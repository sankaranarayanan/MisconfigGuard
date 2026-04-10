"""Regex-based prompt injection detection for pipeline snippets."""

from __future__ import annotations

import re
from typing import Dict, List


class InjectionDetector:
    """Detect explicit attempts to override or subvert LLM instructions."""

    _PATTERNS = [
        re.compile(r"(?i)ignore\s+previous\s+instructions"),
        re.compile(r"(?i)override\s+system\s+prompt"),
        re.compile(r"(?i)exfiltrate\s+data"),
        re.compile(r"(?i)send\s+secrets\s+to\s+external\s+endpoint"),
    ]

    def scan_text(self, text: str, file_path: str = "", base_line: int = 1) -> List[Dict[str, object]]:
        findings: List[Dict[str, object]] = []
        seen = set()
        for offset, line in enumerate(text.splitlines(), 0):
            for pattern in self._PATTERNS:
                if not pattern.search(line):
                    continue
                key = (line.strip().lower(), pattern.pattern)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    {
                        "type": "prompt_injection",
                        "confidence": "high",
                        "severity": "HIGH",
                        "issue": "Malicious instruction detected in pipeline YAML",
                        "explanation": "The pipeline step contains explicit prompt-injection language intended to override prior instructions or exfiltrate sensitive data.",
                        "fix": "Remove the malicious instruction, sanitize untrusted text before it reaches any LLM prompt, and treat pipeline-supplied AI inputs as untrusted.",
                        "line_number": base_line + offset,
                        "line_content": line[:240],
                        "file_path": file_path,
                        "rule_id": "PIPE-PROMPT-001",
                    }
                )
        return findings