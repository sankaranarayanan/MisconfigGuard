"""
WorkloadIdentitySecurityAnalyzer — Hybrid workload identity misconfiguration detection.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, List, Optional

from federation_analyzer import FederationAnalyzer, WorkloadIdentityFinding
from workload_identity_parser import WorkloadIdentityConfig, WorkloadIdentityParser

logger = logging.getLogger(__name__)

_WORKLOAD_IDENTITY_PREAMBLE = """\
You are a cloud identity security analyzer.

STRICT RULES:
- Only analyze the provided workload identity configuration.
- Do not assume missing values or infer claims that are not present.
- If evidence is insufficient, return no issues.
- Return structured JSON only.
"""

_WORKLOAD_IDENTITY_SCHEMA = """\
Return exactly this JSON:

{
  "issues": [
    {
      "identity": "<identity name>",
      "type": "workload_identity",
      "severity": "HIGH | MEDIUM | LOW",
      "issue": "<short issue title>",
      "explanation": "<why this is risky>",
      "fix": "<concrete remediation>",
      "issuer": "<issuer or empty string>",
      "subject": "<subject or empty string>"
    }
  ],
  "summary": "<1-2 sentence assessment>"
}

If no issues are found:
  {"issues": [], "summary": "No workload identity misconfigurations detected."}
"""


def _build_workload_identity_prompt(
    configs: List[WorkloadIdentityConfig],
    query: str,
    retrieved_code_chunks: Optional[List[dict]] = None,
    security_references: Optional[List[dict]] = None,
) -> str:
    lines = [_WORKLOAD_IDENTITY_PREAMBLE, "", "## Workload Identity Configurations"]
    for index, config in enumerate(configs[:15], 1):
        lines.append(
            f"\n### [{index}] Block: {config.block_name} (file: {config.file_path})\n"
            f"```\n{config.source_text[:500]}\n```\n"
            f"Parsed — identity: `{config.identity}`, issuer: `{config.issuer or 'missing'}`, "
            f"subject: `{config.subject or 'missing'}`, audiences: `{', '.join(config.audiences) or 'missing'}`"
        )

    if retrieved_code_chunks:
        lines += ["", "## Retrieved Code Context"]
        for index, chunk in enumerate(retrieved_code_chunks[:5], 1):
            payload = chunk.get("chunk", chunk)
            text = payload.get("text", payload.get("content", ""))
            file_path = payload.get("file_path", chunk.get("file_path", ""))
            if not text:
                continue
            lines.append(f"\n### [Code {chunk.get('rank', index)}] {file_path}\n```\n{text[:500]}\n```")

    if security_references:
        lines += ["", "## Security Rules & Best Practices"]
        for index, rule in enumerate(security_references[:5], 1):
            severity = rule.get("severity", "")
            severity_suffix = f" ({severity})" if severity else ""
            lines.append(
                f"\n### [Rule {index}] {rule.get('title', 'Rule')}{severity_suffix}\n"
                f"{(rule.get('text') or rule.get('description') or '')[:600]}"
            )

    lines += ["", f"## Analysis Query\n{query}", "", "## Instructions", _WORKLOAD_IDENTITY_SCHEMA]
    return "\n".join(lines)


class WorkloadIdentitySecurityAnalyzer:
    def __init__(
        self,
        pipeline: Optional[Any] = None,
        use_llm: bool = True,
        top_k_code: int = 5,
        top_k_security: int = 3,
    ) -> None:
        self.parser = WorkloadIdentityParser()
        self.analyzer = FederationAnalyzer()
        self.pipeline = pipeline
        self.use_llm = use_llm and pipeline is not None
        self.top_k_code = top_k_code
        self.top_k_security = top_k_security

    def analyze_file(self, file_path: str, query: str = "") -> dict:
        if not Path(file_path).is_file():
            raise FileNotFoundError(file_path)
        configs = self.parser.parse_file(file_path)
        return self._run_analysis(
            configs=configs,
            query=query or f"Detect workload identity misconfigurations in {file_path}",
            files=[file_path],
        )

    def analyze_directory(self, directory: str, query: str = "") -> dict:
        if not Path(directory).is_dir():
            raise FileNotFoundError(directory)
        configs: List[WorkloadIdentityConfig] = []
        files: List[str] = []
        for path in Path(directory).rglob("*"):
            if path.suffix.lower() in {".tf", ".yaml", ".yml", ".json"}:
                found = self.parser.parse_file(str(path))
                if found:
                    configs.extend(found)
                    files.append(str(path))
        return self._run_analysis(
            configs=configs,
            query=query or f"Detect workload identity misconfigurations in {directory}",
            files=files,
        )

    def analyze_chunks(self, chunks: List[dict], query: str = "Detect workload identity misconfigurations") -> dict:
        configs = self.parser.parse_chunks(chunks)
        files = sorted({chunk.get("file_path", "") for chunk in chunks if chunk.get("file_path")})
        return self._run_analysis(configs=configs, query=query, files=files)

    def _run_analysis(self, configs: List[WorkloadIdentityConfig], query: str, files: List[str]) -> dict:
        rule_findings = self.analyzer.analyze(configs)
        llm_response = ""
        code_chunks: List[dict] = []
        security_refs: List[dict] = []

        if self.use_llm and configs:
            try:
                llm_response, code_chunks, security_refs = self._llm_enrich(configs, query)
            except Exception as exc:
                logger.warning("Workload identity LLM enrichment failed: %s", exc)

        issues = self._merge_findings(rule_findings, llm_response)
        summary = self._build_summary(issues, configs, files)
        return {
            "issues": [issue.to_dict() for issue in issues],
            "summary": summary,
            "evidence": {
                "code_chunks": code_chunks,
                "security_references": security_refs,
            },
            "analysis": llm_response,
            "metadata": {
                "total_configs": len(configs),
                "files_analyzed": sorted(set(files)),
                "rule_findings_count": len(rule_findings),
                "llm_used": bool(llm_response),
            },
        }

    def _llm_enrich(self, configs: List[WorkloadIdentityConfig], query: str):
        from rag_orchestrator import RAGOrchestrator

        llm_response = ""
        code_chunks: List[dict] = []
        security_refs: List[dict] = []
        vector_store = getattr(self.pipeline, "vector_store", None)
        has_code_index = bool(getattr(vector_store, "total_vectors", 0) > 0)

        security_kb = None
        if hasattr(self.pipeline, "embedder"):
            try:
                from security_kb import SecurityKnowledgeBase

                security_kb = SecurityKnowledgeBase(embedder=self.pipeline.embedder)
                security_kb.load_or_build()
            except Exception as exc:
                logger.info("Workload identity analyzer: could not load security KB: %s", exc)

        if not has_code_index and security_kb is None:
            return llm_response, code_chunks, security_refs

        orchestrator = RAGOrchestrator.from_pipeline(
            pipeline=self.pipeline,
            security_kb=security_kb,
            top_k_code=self.top_k_code,
            top_k_security=self.top_k_security,
        )

        workload_query = (
            "Analyze workload identity federation configuration. Flag invalid issuer, broad subject, "
            f"missing audience, and trust misconfigurations. Only use provided data. {query}"
        )

        if has_code_index:
            results = orchestrator._retrieve_code(workload_query, self.top_k_code, {"cloud_provider": "azure"})
            code_chunks = [result.to_dict() for result in results]

        security_results = orchestrator._retrieve_security_rules(workload_query, self.top_k_security, [])
        security_refs = [
            result.to_dict() if hasattr(result, "to_dict") else result
            for result in security_results
        ]

        if not code_chunks and not security_refs:
            return llm_response, code_chunks, security_refs

        prompt = _build_workload_identity_prompt(
            configs=configs,
            query=workload_query,
            retrieved_code_chunks=code_chunks,
            security_references=security_refs,
        )
        if hasattr(self.pipeline, "llm_client") and self.pipeline.llm_client.is_available():
            llm_response = self.pipeline.llm_client.generate(prompt)
        return llm_response, code_chunks, security_refs

    def _merge_findings(self, rule_findings: List[WorkloadIdentityFinding], llm_response: str) -> List[WorkloadIdentityFinding]:
        llm_issues = self._parse_llm_issues(llm_response)
        seen = {(finding.identity.lower(), finding.issue.lower()) for finding in rule_findings}
        merged = list(rule_findings)
        for issue in llm_issues:
            key = (issue.get("identity", "").lower(), issue.get("issue", "").lower())
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                WorkloadIdentityFinding(
                    rule_id="WID-LLM",
                    identity=issue.get("identity", ""),
                    type=issue.get("type", "workload_identity"),
                    severity=issue.get("severity", "LOW").upper(),
                    issue=issue.get("issue", ""),
                    explanation=issue.get("explanation", ""),
                    fix=issue.get("fix", ""),
                    issuer=issue.get("issuer", ""),
                    subject=issue.get("subject", ""),
                    audiences=issue.get("audiences", []),
                )
            )
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        merged.sort(key=lambda item: order.get(item.severity, 9))
        return merged

    @staticmethod
    def _parse_llm_issues(raw: str) -> List[dict]:
        if not raw:
            return []
        try:
            return json.loads(raw.strip()).get("issues", [])
        except (json.JSONDecodeError, ValueError):
            pass
        match = re.search(r"\{[\s\S]+\}", raw)
        if match:
            try:
                return json.loads(match.group()).get("issues", [])
            except (json.JSONDecodeError, ValueError):
                return []
        return []

    @staticmethod
    def _build_summary(
        issues: List[WorkloadIdentityFinding],
        configs: List[WorkloadIdentityConfig],
        files: List[str],
    ) -> dict:
        return {
            "total_configs": len(configs),
            "total_findings": len(issues),
            "high_severity": sum(1 for issue in issues if issue.severity == "HIGH"),
            "medium_severity": sum(1 for issue in issues if issue.severity == "MEDIUM"),
            "low_severity": sum(1 for issue in issues if issue.severity == "LOW"),
            "files_analyzed": sorted(set(files)),
        }