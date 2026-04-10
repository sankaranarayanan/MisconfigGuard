"""Optional local-LLM validation for prompt-injection findings."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VALIDATOR_PREAMBLE = """\
You are a CI/CD security analyzer.

STRICT RULES:
- Only evaluate the provided pipeline snippet and findings.
- Do not infer hidden commands or data flows.
- Do not invent new issues without direct evidence from the snippet.
- Return structured JSON only.
"""

_VALIDATOR_SCHEMA = """\
Return exactly this JSON:

{
  "issues": [
    {
      "file_path": "<source file>",
      "type": "prompt_injection | script_injection | external_input",
      "severity": "HIGH | MEDIUM | LOW",
      "issue": "<short issue title>",
      "explanation": "<why this is risky>",
      "fix": "<concrete remediation>",
      "line_number": 0
    }
  ],
  "summary": "<1-2 sentence assessment>"
}

If no issues are found:
  {"issues": [], "summary": "No prompt injection issues detected."}
"""


class LLMValidator:
    """Use the local pipeline + KB to validate prompt-injection detections."""

    def __init__(self, pipeline: Optional[Any] = None, top_k_security: int = 3) -> None:
        self.pipeline = pipeline
        self.top_k_security = top_k_security

    def validate(
        self,
        *,
        file_path: str,
        snippet: str,
        findings: List[Dict[str, Any]],
        query: str,
    ) -> Tuple[str, List[Dict[str, Any]], List[dict]]:
        if self.pipeline is None or not hasattr(self.pipeline, "llm_client"):
            return "", [], []
        if not self.pipeline.llm_client.is_available():
            return "", [], []

        security_refs: List[dict] = []
        security_kb = None
        if hasattr(self.pipeline, "embedder"):
            try:
                from security_kb import SecurityKnowledgeBase

                security_kb = SecurityKnowledgeBase(embedder=self.pipeline.embedder)
                security_kb.load_or_build()
            except Exception as exc:
                logger.info("Prompt injection validator: could not load security KB: %s", exc)

        if security_kb is not None:
            try:
                from rag_orchestrator import RAGOrchestrator

                orchestrator = RAGOrchestrator.from_pipeline(
                    pipeline=self.pipeline,
                    security_kb=security_kb,
                    top_k_code=0,
                    top_k_security=self.top_k_security,
                )
                security_results = orchestrator._retrieve_security_rules(query, self.top_k_security, [])
                security_refs = [item.to_dict() if hasattr(item, "to_dict") else item for item in security_results]
            except Exception as exc:
                logger.info("Prompt injection validator: could not retrieve security rules: %s", exc)

        prompt = self._build_prompt(file_path=file_path, snippet=snippet, findings=findings, query=query, security_refs=security_refs)
        raw = self.pipeline.llm_client.generate(prompt)
        return raw, self._parse_issues(raw), security_refs

    def _build_prompt(
        self,
        *,
        file_path: str,
        snippet: str,
        findings: List[Dict[str, Any]],
        query: str,
        security_refs: List[dict],
    ) -> str:
        lines = [_VALIDATOR_PREAMBLE, "", f"## File\n{file_path}", "", "## Pipeline Snippet", f"```yaml\n{snippet[:1200]}\n```", "", "## Rule-Based Findings", json.dumps(findings[:10], indent=2)]
        if security_refs:
            lines += ["", "## Security Rules"]
            for index, rule in enumerate(security_refs[:5], 1):
                lines.append(f"\n### [Rule {index}] {rule.get('title', 'Rule')}\n{(rule.get('text') or rule.get('description') or '')[:500]}")
        lines += ["", f"## Query\n{query}", "", "## Instructions", _VALIDATOR_SCHEMA]
        return "\n".join(lines)

    def _parse_issues(self, raw: str) -> List[Dict[str, Any]]:
        if not raw:
            return []
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        issues = payload.get("issues", [])
        return issues if isinstance(issues, list) else []