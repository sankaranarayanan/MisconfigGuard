"""
HardcodedSecretsAnalyzer — hybrid hardcoded secret detection for IaC files.

Combines regex and entropy-based detection with optional LLM enrichment using
the existing code index and security knowledge base.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from secret_scanner import SecretMatch, SecretScanner

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".tf", ".yaml", ".yml", ".json"}

_SECRETS_PREAMBLE = """\
You are a cloud security analyzer.

STRICT RULES:
- Only analyze the provided findings and snippets.
- Do not invent secrets or assume values not present in the evidence.
- Do not repeat raw secret values.
- Return structured JSON only.
"""

_SECRETS_SCHEMA = """\
Return exactly this JSON:

{
  "issues": [
    {
      "file_path": "<source file>",
      "secret_type": "<secret type>",
      "severity": "HIGH | MEDIUM | LOW",
      "issue": "<short issue title>",
      "explanation": "<why this is risky>",
      "fix": "<concrete remediation>"
    }
  ],
  "summary": "<1-2 sentence assessment>"
}

If no issues are found:
  {"issues": [], "summary": "No hardcoded secrets detected."}
"""

_FIX_BY_TYPE = {
    "password": "Remove the hardcoded password and store it in Azure Key Vault or another managed secret store. Inject it at runtime through environment variables or workload identity.",
    "aws_key": "Remove hardcoded AWS credentials. Use IAM roles or short-lived credentials, and store any required bootstrap secret in Azure Key Vault or AWS Secrets Manager instead of source control.",
    "aws_secret": "Delete the embedded AWS secret access key, rotate it immediately, and replace it with IAM role-based access or a secret reference from Azure Key Vault or AWS Secrets Manager.",
    "azure_storage_key": "Replace the storage account key with Azure Key Vault or Managed Identity. Rotate the exposed key and avoid embedding account keys in templates.",
    "azure_connection_string": "Move the connection string into Azure Key Vault and prefer Managed Identity over connection strings where the platform supports it.",
    "api_key": "Store the API key in Azure Key Vault or a platform secret manager, rotate the leaked key, and inject it through configuration at deploy time.",
    "token": "Remove the hardcoded token, rotate it, and load replacement credentials from Azure Key Vault or another managed secret store.",
    "secret": "Move the secret value into Azure Key Vault or a managed secret store and reference it at runtime instead of committing it to code.",
    "connection_string": "Do not commit connection strings. Store them in Azure Key Vault and prefer identity-based connections where possible.",
    "private_key": "Remove the embedded private key immediately, rotate the keypair, and store private keys in Azure Key Vault, HSM-backed storage, or a secure secret manager.",
    "certificate": "Store certificate material outside source control, ideally in Azure Key Vault or a certificate manager, and rotate if the committed certificate includes private material.",
    "jwt_token": "Do not hardcode JWTs in code. Rotate the token and fetch runtime credentials from Azure Key Vault or a secure token service.",
    "high_entropy": "Review this high-entropy value to confirm whether it is a secret. If it is, move it to Azure Key Vault or another managed secret store and rotate it if exposed.",
}


def _fix_for(secret_type: str) -> str:
    return _FIX_BY_TYPE.get(
        secret_type,
        "Move the secret out of source control and into Azure Key Vault or another managed secret store, then rotate the exposed credential.",
    )


def _issue_for(secret_type: str) -> str:
    labels = {
        "aws_key": "Hardcoded AWS access key",
        "aws_secret": "Hardcoded AWS secret access key",
        "azure_storage_key": "Hardcoded Azure storage account key",
        "azure_connection_string": "Hardcoded Azure connection string",
        "password": "Hardcoded password",
        "api_key": "Hardcoded API key",
        "token": "Hardcoded access token",
        "secret": "Hardcoded secret value",
        "connection_string": "Hardcoded connection string",
        "private_key": "Hardcoded private key material",
        "certificate": "Embedded certificate material",
        "jwt_token": "Hardcoded JWT token",
        "high_entropy": "High-entropy value may be a secret",
    }
    return labels.get(secret_type, "Hardcoded secret detected")


def _explanation_for(secret_type: str) -> str:
    explanations = {
        "high_entropy": "This value looks like a generated credential or token. Even if the exact type is unclear, committing high-entropy secrets to source control increases exposure and rotation cost.",
        "private_key": "Private key material in source control can be copied and reused by anyone with repository access, enabling impersonation and long-lived compromise.",
        "azure_connection_string": "Connection strings often embed account credentials. Storing them in code expands the blast radius of repository access and makes secret rotation harder.",
    }
    return explanations.get(
        secret_type,
        "Secrets embedded directly in infrastructure code can be exposed through source control, logs, build artifacts, and developer workstations. They should be stored in a managed secret store instead.",
    )


def _build_secrets_prompt(
    findings: List[Dict[str, Any]],
    query: str,
    snippets: Optional[List[dict]] = None,
    security_references: Optional[List[dict]] = None,
) -> str:
    lines = [_SECRETS_PREAMBLE, "", "## Detected Secret Findings"]

    for index, finding in enumerate(findings[:20], 1):
        lines.append(
            f"\n### [{index}] {finding.get('file_path', '') or 'unknown file'}:{finding.get('line_number', 0)}\n"
            f"type: `{finding.get('secret_type', '')}`\n"
            f"severity: `{finding.get('severity', '')}`\n"
            f"confidence: `{finding.get('confidence', '')}`\n"
            f"masked match: `{finding.get('match', '')}`\n"
            f"context:\n```\n{finding.get('line_content', '')[:300]}\n```"
        )

    if snippets:
        lines += ["", "## Retrieved Code Context"]
        for index, chunk in enumerate(snippets[:5], 1):
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

    lines += ["", f"## Analysis Query\n{query}", "", "## Instructions", _SECRETS_SCHEMA]
    return "\n".join(lines)


class HardcodedSecretsAnalyzer:
    def __init__(
        self,
        pipeline: Optional[Any] = None,
        use_llm: bool = True,
        top_k_code: int = 5,
        top_k_security: int = 3,
        entropy_threshold: float = 4.5,
    ) -> None:
        self.scanner = SecretScanner(entropy_threshold=entropy_threshold)
        self.pipeline = pipeline
        self.use_llm = use_llm and pipeline is not None
        self.top_k_code = top_k_code
        self.top_k_security = top_k_security

    def analyze_file(self, file_path: str, query: str = "") -> dict:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(file_path)
        findings = [self._match_to_finding(match) for match in self.scanner.scan_file(str(path))]
        return self._run_analysis(
            findings=findings,
            query=query or f"Detect hardcoded secrets in {file_path}",
            files=[str(path)],
        )

    def analyze_directory(self, directory: str, query: str = "") -> dict:
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(directory)

        findings: List[Dict[str, Any]] = []
        files: List[str] = []
        for path in root.rglob("*"):
            if path.suffix.lower() not in _SUPPORTED_SUFFIXES or not path.is_file():
                continue
            files.append(str(path))
            file_findings = [self._match_to_finding(match) for match in self.scanner.scan_file(str(path))]
            if file_findings:
                findings.extend(file_findings)

        return self._run_analysis(
            findings=findings,
            query=query or f"Detect hardcoded secrets in {directory}",
            files=files,
        )

    def analyze_chunks(self, chunks: List[dict], query: str = "Detect hardcoded secrets") -> dict:
        findings: List[Dict[str, Any]] = []
        files = sorted({chunk.get("file_path", "") for chunk in chunks if chunk.get("file_path")})

        for chunk in chunks:
            text = chunk.get("text") or chunk.get("content") or chunk.get("chunk", {}).get("text", "")
            file_path = chunk.get("file_path") or chunk.get("chunk", {}).get("file_path", "")
            for match in self.scanner.scan_text(text or "", file_path=file_path):
                findings.append(self._match_to_finding(match))

        return self._run_analysis(findings=findings, query=query, files=files)

    def _run_analysis(self, findings: List[Dict[str, Any]], query: str, files: List[str]) -> dict:
        llm_response = ""
        code_chunks: List[dict] = []
        security_refs: List[dict] = []

        if self.use_llm and findings:
            try:
                llm_response, code_chunks, security_refs = self._llm_enrich(findings, query)
            except Exception as exc:
                logger.warning("Secrets LLM enrichment failed: %s", exc)

        merged_issues = self._merge_findings(findings, llm_response)
        summary = self._build_summary(merged_issues, files)
        return {
            "issues": merged_issues,
            "summary": summary,
            "evidence": {
                "code_chunks": code_chunks,
                "security_references": security_refs,
            },
            "analysis": llm_response,
            "metadata": {
                "files_analyzed": sorted(set(files)),
                "scan_findings_count": len(findings),
                "llm_used": bool(llm_response),
            },
        }

    def _llm_enrich(self, findings: List[Dict[str, Any]], query: str):
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
                logger.info("Secrets analyzer: could not load security KB: %s", exc)

        if not has_code_index and security_kb is None:
            return llm_response, code_chunks, security_refs

        orchestrator = RAGOrchestrator.from_pipeline(
            pipeline=self.pipeline,
            security_kb=security_kb,
            top_k_code=self.top_k_code,
            top_k_security=self.top_k_security,
        )

        secrets_query = (
            "Analyze the provided findings for hardcoded passwords, API keys, tokens, "
            "private keys, connection strings, and other embedded credentials. "
            f"Only use provided evidence. {query}"
        )

        if has_code_index:
            results = orchestrator._retrieve_code(secrets_query, self.top_k_code, {})
            code_chunks = [result.to_dict() for result in results]

        security_results = orchestrator._retrieve_security_rules(secrets_query, self.top_k_security, [])
        security_refs = [
            result.to_dict() if hasattr(result, "to_dict") else result
            for result in security_results
        ]

        if not code_chunks and not security_refs:
            return llm_response, code_chunks, security_refs

        prompt = _build_secrets_prompt(
            findings=findings,
            query=secrets_query,
            snippets=code_chunks,
            security_references=security_refs,
        )
        if hasattr(self.pipeline, "llm_client") and self.pipeline.llm_client.is_available():
            llm_response = self.pipeline.llm_client.generate(prompt)
        return llm_response, code_chunks, security_refs

    def _merge_findings(self, rule_findings: List[Dict[str, Any]], llm_response: str) -> List[Dict[str, Any]]:
        merged = list(rule_findings)
        seen = {
            (item.get("file_path", ""), item.get("secret_type", ""), item.get("line_number", 0), item.get("issue", ""))
            for item in merged
        }

        for issue in self._parse_llm_issues(llm_response):
            key = (
                issue.get("file_path", ""),
                issue.get("secret_type", ""),
                issue.get("line_number", 0),
                issue.get("issue", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            issue.setdefault("confidence", "medium")
            issue.setdefault("rule_id", "SECRET-LLM")
            issue.setdefault("match", "")
            issue.setdefault("line_content", "")
            issue.setdefault("line_number", 0)
            issue.setdefault("explanation", _explanation_for(issue.get("secret_type", "")))
            issue.setdefault("fix", _fix_for(issue.get("secret_type", "")))
            merged.append(issue)

        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        merged.sort(key=lambda item: (order.get(item.get("severity", "LOW"), 9), item.get("file_path", ""), item.get("line_number", 0)))
        return merged

    def _parse_llm_issues(self, llm_response: str) -> List[Dict[str, Any]]:
        if not llm_response:
            return []
        match = re.search(r"\{.*\}", llm_response, re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.debug("Secrets analyzer received non-JSON LLM response")
            return []
        issues = payload.get("issues", [])
        return issues if isinstance(issues, list) else []

    def _match_to_finding(self, match: SecretMatch) -> Dict[str, Any]:
        return {
            "rule_id": f"SECRET-{match.secret_type.upper()}",
            "file_path": match.file_path,
            "secret_type": match.secret_type,
            "severity": match.severity,
            "issue": _issue_for(match.secret_type),
            "explanation": _explanation_for(match.secret_type),
            "fix": _fix_for(match.secret_type),
            "line_number": match.line_number,
            "line_content": match.line_content,
            "match": match.match,
            "confidence": match.confidence,
        }

    def _build_summary(self, issues: List[Dict[str, Any]], files: List[str]) -> Dict[str, Any]:
        high = sum(1 for issue in issues if issue.get("severity") == "HIGH")
        medium = sum(1 for issue in issues if issue.get("severity") == "MEDIUM")
        low = sum(1 for issue in issues if issue.get("severity") == "LOW")
        return {
            "total_matches": len(issues),
            "total_findings": len(issues),
            "files_analyzed": sorted(set(files)),
            "high_severity": high,
            "medium_severity": medium,
            "low_severity": low,
        }