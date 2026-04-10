"""
IAMSecurityAnalyzer — Top-level coordinator for managed identity over-permission detection.

Combines:
    1. IAMParser          — extract role assignments from Terraform / ARM templates
    2. PermissionAnalyzer — rule-based detection (always runs, no LLM required)
    3. RAGOrchestrator    — optional LLM enrichment with security-rule context

The LLM step uses a tightly-scoped IAM prompt that instructs the model to:
    • Only use provided IAM data (no inference from general knowledge)
    • Base findings on CIS Azure Benchmark and least-privilege principles
    • Return structured JSON findings

Output schema (always returned)
--------------------------------
{
    "issues": [
        {
            "rule_id":        "IAM-AZ-001",
            "identity":       "...",
            "role":           "Contributor",
            "scope":          "subscription",
            "severity":       "HIGH",
            "issue":          "...",
            "explanation":    "...",
            "fix":            "...",
            "recommendations": [...],
            "file_path":      "...",
            "block_name":     "...",
        }
    ],
    "summary": {
        "total_assignments": N,
        "total_findings":    N,
        "high_severity":     N,
        "medium_severity":   N,
        "low_severity":      N,
        "files_analyzed":    [...],
    },
    "evidence": {
        "code_chunks":         [...],
        "security_references": [...],
    },
    "analysis":  str,   # raw LLM response (empty string if LLM not used)
    "metadata": {
        "total_assignments":   N,
        "files_analyzed":      [...],
        "rule_findings_count": N,
        "llm_used":            bool,
    },
}

Usage
-----
    # Rule-based only (no Ollama required)
    analyzer = IAMSecurityAnalyzer()
    result = analyzer.analyze_file("main.tf")

    # With LLM enrichment
    analyzer = IAMSecurityAnalyzer(pipeline=rag_pipeline)
    result = analyzer.analyze_directory("./terraform/")
    for issue in result["issues"]:
        print(f"[{issue['severity']}] {issue['identity']}: {issue['issue']}")
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from iam_parser import IAMParser, RoleAssignment
from permission_analyzer import IAMFinding, PermissionAnalyzer, RecommendationEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IAM-specific LLM prompt template
# ---------------------------------------------------------------------------

_IAM_SYSTEM_PREAMBLE = """\
You are an expert Azure cloud security engineer specialising in Identity and
Access Management (IAM).

STRICT RULES:
- Analyse ONLY the role assignment data provided below.
- Do NOT infer permissions that are not explicitly present in the data.
- Base all findings on: CIS Azure Benchmark, Microsoft least-privilege
  principles, and Azure RBAC best practices.
- Return ONLY valid JSON — no markdown fences, no preamble.\
"""

_IAM_OUTPUT_SCHEMA = """\
Return exactly this JSON (no extra keys, no comments):

{
  "issues": [
    {
      "identity":      "<identity name>",
      "role":          "<role display name>",
      "scope":         "subscription | resource_group | resource | unknown",
      "severity":      "HIGH | MEDIUM | LOW",
      "issue":         "<short issue title (≤ 80 chars)>",
      "explanation":   "<why this is an over-permission risk>",
      "fix":           "<concrete remediation with example role/scope>",
      "cis_reference": "<CIS Azure X.Y or empty string>"
    }
  ],
  "summary": "<1-2 sentence overall assessment>"
}

If no issues are found:
  {"issues": [], "summary": "No managed identity over-permission issues detected."}\
"""


def _build_iam_prompt(
    assignments: List[RoleAssignment],
    query: str,
    retrieved_code_chunks: Optional[List[dict]] = None,
    security_references: Optional[List[dict]] = None,
) -> str:
    """Build a focused IAM-analysis prompt for the local LLM."""
    lines = [
        _IAM_SYSTEM_PREAMBLE,
        "",
        "## Role Assignments to Analyse",
    ]
    # Cap at 15 assignments to stay within typical context windows
    for i, a in enumerate(assignments[:15], 1):
        lines.append(
            f"\n### [{i}] Block: {a.block_name}  (file: {a.file_path})\n"
            f"```hcl\n{a.source_text[:500]}\n```\n"
            f"Parsed — identity: `{a.identity_name}`, "
            f"role: `{a.role}`, "
            f"scope: `{a.scope_type}` (`{a.scope_value or 'reference'}`)"
        )
    if len(assignments) > 15:
        lines.append(f"\n_(... {len(assignments) - 15} additional assignments truncated)_")

    if retrieved_code_chunks:
        lines += ["", "## Retrieved Code Context"]
        for index, chunk in enumerate(retrieved_code_chunks[:5], 1):
            payload = chunk.get("chunk", chunk)
            file_path = payload.get("file_path", chunk.get("file_path", ""))
            text = payload.get("text", payload.get("content", ""))
            rank = chunk.get("rank", index)
            if not text:
                continue
            lines.append(
                f"\n### [Code {rank}] {file_path or 'retrieved chunk'}\n"
                f"```\n{text[:500]}\n```"
            )

    if security_references:
        lines += ["", "## Security Rules & Best Practices"]
        for index, rule in enumerate(security_references[:5], 1):
            title = rule.get("title", f"Rule {index}")
            severity = rule.get("severity", "")
            text = rule.get("text") or rule.get("description") or ""
            lines.append(
                f"\n### [Rule {index}] {title}"
                f"{f' ({severity})' if severity else ''}\n{text[:600]}"
            )

    lines += [
        "",
        f"## Analysis Query\n{query}",
        "",
        "## Instructions",
        _IAM_OUTPUT_SCHEMA,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# IAMSecurityAnalyzer
# ---------------------------------------------------------------------------


class IAMSecurityAnalyzer:
    """
    Detect managed identity over-permission issues in Azure IaC configurations.

    Combines rule-based detection (always runs) with optional LLM enrichment
    when a :class:`~rag_pipeline.RAGPipeline` is provided.

    Parameters
    ----------
    pipeline :
        An initialised ``RAGPipeline`` for LLM enrichment and context
        retrieval.  If ``None``, only rule-based detection is performed.
    use_llm :
        Whether to call the local LLM for additional reasoning.
        Automatically disabled when *pipeline* is ``None``.
    max_roles_per_identity :
        Threshold for the "too many roles" detection rule (default 3).
    top_k_code :
        Number of code chunks to retrieve for LLM context (default 5).
    top_k_security :
        Number of security rules to retrieve from the KB (default 3).
    """

    def __init__(
        self,
        pipeline: Optional[Any] = None,
        use_llm: bool = True,
        max_roles_per_identity: int = 3,
        top_k_code: int = 5,
        top_k_security: int = 3,
    ) -> None:
        self.parser    = IAMParser()
        self.analyzer  = PermissionAnalyzer(
            max_roles_per_identity=max_roles_per_identity,
        )
        self.rec       = RecommendationEngine()
        self.pipeline  = pipeline
        self.use_llm   = use_llm and pipeline is not None
        self.top_k_code     = top_k_code
        self.top_k_security = top_k_security

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def analyze_file(
        self,
        file_path: str,
        query: str = "",
    ) -> dict:
        """Analyse a single Terraform / YAML / JSON file."""
        assignments = self.parser.parse_file(file_path)
        return self._run_analysis(
            assignments = assignments,
            query       = query or f"Detect managed identity over-permissions in {file_path}",
            files       = [file_path],
        )

    def analyze_directory(
        self,
        directory: str,
        query: str = "",
    ) -> dict:
        """Recursively analyse all IaC files under *directory*."""
        assignments: List[RoleAssignment] = []
        files_analyzed: List[str]         = []

        for path in Path(directory).rglob("*"):
            if path.suffix.lower() in (".tf", ".yaml", ".yml", ".json"):
                found = self.parser.parse_file(str(path))
                if found:
                    assignments.extend(found)
                    files_analyzed.append(str(path))

        return self._run_analysis(
            assignments = assignments,
            query       = query or f"Detect managed identity over-permissions in {directory}",
            files       = files_analyzed,
        )

    def analyze_chunks(
        self,
        chunks: List[dict],
        query: str = "Detect managed identity over-permissions",
    ) -> dict:
        """Analyse role assignments from pre-chunked pipeline data."""
        assignments  = self.parser.parse_chunks(chunks)
        file_paths   = list({c.get("file_path", "") for c in chunks if c.get("file_path")})
        return self._run_analysis(
            assignments = assignments,
            query       = query,
            files       = file_paths,
        )

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _run_analysis(
        self,
        assignments: List[RoleAssignment],
        query: str,
        files: List[str],
    ) -> dict:
        """Rule-based detection + optional LLM enrichment → structured output."""
        # 1. Rule-based detection (always runs, no LLM needed)
        rule_findings = self.analyzer.analyze(assignments)

        # 2. Optional LLM enrichment
        llm_response  = ""
        code_chunks:   List[dict] = []
        security_refs: List[dict] = []

        if self.use_llm and assignments:
            try:
                llm_response, code_chunks, security_refs = self._llm_enrich(
                    assignments, query
                )
            except Exception as exc:
                logger.warning("LLM enrichment failed: %s", exc)

        # 3. Merge rule-based + LLM findings (deduplicate)
        issues = self._merge_findings(rule_findings, llm_response)

        # 4. Build summary stats
        summary = self._build_summary(issues, assignments, files)

        return {
            "issues":   [f.to_dict() for f in issues],
            "summary":  summary,
            "evidence": {
                "code_chunks":         code_chunks,
                "security_references": security_refs,
            },
            "analysis": llm_response,
            "metadata": {
                "total_assignments":   len(assignments),
                "files_analyzed":      sorted(set(files)),
                "rule_findings_count": len(rule_findings),
                "llm_used":            bool(llm_response),
            },
        }

    # ------------------------------------------------------------------
    # LLM enrichment
    # ------------------------------------------------------------------

    def _llm_enrich(
        self,
        assignments: List[RoleAssignment],
        query: str,
    ):
        """Call the RAG pipeline for LLM-enriched IAM analysis."""
        from rag_orchestrator import RAGOrchestrator

        code_chunks:   List[dict] = []
        security_refs: List[dict] = []
        llm_response = ""

        iam_q = (
            "Analyze Azure role assignments. Flag any managed identity with excessive "
            f"permissions. Suggest least-privilege alternatives. Only use provided data. {query}"
        )

        vector_store = getattr(self.pipeline, "vector_store", None)
        has_code_index = bool(getattr(vector_store, "total_vectors", 0) > 0)

        security_kb = None
        if hasattr(self.pipeline, "embedder"):
            try:
                from security_kb import SecurityKnowledgeBase

                security_kb = SecurityKnowledgeBase(embedder=self.pipeline.embedder)
                security_kb.load_or_build()
            except Exception as exc:
                logger.info("IAMSecurityAnalyzer: could not load security KB: %s", exc)
                security_kb = None

        if not has_code_index and security_kb is None:
            logger.info(
                "IAMSecurityAnalyzer: no retrievable IAM context available — skipping LLM step"
            )
            return llm_response, code_chunks, security_refs

        orch = RAGOrchestrator.from_pipeline(
            pipeline=self.pipeline,
            security_kb=security_kb,
            top_k_code=self.top_k_code,
            top_k_security=self.top_k_security,
        )

        if has_code_index:
            code_results = orch._retrieve_code(
                iam_q,
                self.top_k_code,
                {"cloud_provider": "azure"},
            )
            code_chunks = [result.to_dict() for result in code_results]

        security_results = orch._retrieve_security_rules(
            iam_q,
            self.top_k_security,
            [],
        )
        security_refs = [
            result.to_dict() if hasattr(result, "to_dict") else result
            for result in security_results
        ]

        if not code_chunks and not security_refs:
            logger.info(
                "IAMSecurityAnalyzer: retrieval returned no IAM evidence — skipping LLM step"
            )
            return llm_response, code_chunks, security_refs

        prompt = _build_iam_prompt(
            assignments=assignments,
            query=iam_q,
            retrieved_code_chunks=code_chunks,
            security_references=security_refs,
        )
        if hasattr(self.pipeline, "llm_client"):
            if self.pipeline.llm_client.is_available():
                llm_response = self.pipeline.llm_client.generate(prompt)
            else:
                logger.info(
                    "IAMSecurityAnalyzer: Ollama unavailable — skipping LLM step"
                )

        return llm_response, code_chunks, security_refs

    # ------------------------------------------------------------------
    # Merge & dedup
    # ------------------------------------------------------------------

    def _merge_findings(
        self,
        rule_findings: List[IAMFinding],
        llm_response:  str,
    ) -> List[IAMFinding]:
        """
        Merge rule-based findings with any new issues extracted from the LLM.

        Rule-based findings always take priority.  LLM findings are appended
        only when they report a (identity, role, scope) combination not already
        covered by the rule engine.
        """
        llm_issues = self._parse_llm_issues(llm_response)

        # Build dedup key set from rule findings
        seen = {
            (f.identity.lower(), f.role.lower(), f.scope)
            for f in rule_findings
        }
        merged = list(rule_findings)

        for li in llm_issues:
            key = (
                li.get("identity", "").lower(),
                li.get("role",     "").lower(),
                li.get("scope",    ""),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                IAMFinding(
                    rule_id        = "IAM-LLM",
                    identity       = li.get("identity",    ""),
                    role           = li.get("role",        ""),
                    scope          = li.get("scope",       "unknown"),
                    scope_value    = "",
                    severity       = li.get("severity",    "LOW").upper(),
                    issue          = li.get("issue",       ""),
                    explanation    = li.get("explanation", ""),
                    fix            = li.get("fix",         ""),
                    recommendations = [],
                    file_path      = "",
                    block_name     = "",
                )
            )

        # Sort: HIGH → MEDIUM → LOW
        _order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        merged.sort(key=lambda f: _order.get(f.severity, 3))
        return merged

    @staticmethod
    def _parse_llm_issues(raw: str) -> List[dict]:
        """Extract the ``issues`` list from an LLM JSON response."""
        if not raw:
            return []
        try:
            return json.loads(raw.strip()).get("issues", [])
        except (json.JSONDecodeError, ValueError):
            pass
        # Regex fallback: extract first JSON object block
        m = re.search(r"\{[\s\S]+\}", raw)
        if m:
            try:
                return json.loads(m.group()).get("issues", [])
            except (json.JSONDecodeError, ValueError):
                pass
        return []

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        issues: List[IAMFinding],
        assignments: List[RoleAssignment],
        files: List[str],
    ) -> dict:
        high   = sum(1 for f in issues if f.severity == "HIGH")
        medium = sum(1 for f in issues if f.severity == "MEDIUM")
        low    = sum(1 for f in issues if f.severity == "LOW")
        return {
            "total_assignments": len(assignments),
            "total_findings":    len(issues),
            "high_severity":     high,
            "medium_severity":   medium,
            "low_severity":      low,
            "files_analyzed":    sorted(set(files)),
        }
