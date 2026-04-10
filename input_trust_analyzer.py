"""Track untrusted inputs flowing into prompts or shell commands."""

from __future__ import annotations

import re
from typing import Dict, List


class InputTrustAnalyzer:
    """Detect direct use of PR metadata or untrusted variables in prompts or commands."""

    _UNTRUSTED = re.compile(
        r"(?i)(github\.event\.pull_request\.(?:title|body)|github\.event\.issue\.(?:title|body)|system\.pullrequest|\bUNTRUSTED_INPUT\b)"
    )
    _SINK = re.compile(r"(?i)(--prompt\b|ollama\b|llm\b|eval\b|bash\b|sh\b|python\b.*--prompt)")

    def scan_text(self, text: str, file_path: str = "", base_line: int = 1) -> List[Dict[str, object]]:
        findings: List[Dict[str, object]] = []
        for offset, line in enumerate(text.splitlines(), 0):
            if not self._UNTRUSTED.search(line):
                continue
            direct_sink = bool(self._SINK.search(line))
            findings.append(
                {
                    "type": "external_input",
                    "confidence": "high" if direct_sink else "medium",
                    "severity": "HIGH" if direct_sink else "MEDIUM",
                    "issue": "Untrusted external input used in pipeline command",
                    "explanation": "The pipeline uses PR-controlled or otherwise untrusted input directly in a command or AI prompt, which can enable prompt injection or command manipulation.",
                    "fix": "Treat PR metadata and external values as tainted input, validate them before use, and avoid passing them directly into shell commands or LLM prompts.",
                    "line_number": base_line + offset,
                    "line_content": line[:240],
                    "file_path": file_path,
                    "rule_id": "PIPE-INPUT-001",
                }
            )
        return findings