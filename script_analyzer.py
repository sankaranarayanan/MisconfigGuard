"""Detect unsafe dynamic command execution in pipeline scripts."""

from __future__ import annotations

import re
from typing import Dict, List


class ScriptAnalyzer:
    """Flag remote script execution and dynamic eval patterns."""

    _REMOTE_SCRIPT = re.compile(r"(?i)\b(?:curl|wget)\b[^\n|;]*(https?://[^\s|;]+)[^\n]*\|\s*(?:bash|sh)\b")
    _EVAL = re.compile(r"(?i)\beval\b\s+(.+)")
    _EXEC = re.compile(r"(?i)\bexec\s*\(")

    def scan_text(self, text: str, file_path: str = "", base_line: int = 1) -> List[Dict[str, object]]:
        findings: List[Dict[str, object]] = []
        for offset, line in enumerate(text.splitlines(), 0):
            remote_match = self._REMOTE_SCRIPT.search(line)
            if remote_match:
                findings.append(
                    {
                        "type": "script_injection",
                        "confidence": "high",
                        "severity": "HIGH",
                        "issue": "Remote script execution in pipeline step",
                        "explanation": "The pipeline downloads a remote script and pipes it directly into a shell, which allows upstream content changes to execute unreviewed code in CI.",
                        "fix": "Pin trusted scripts by digest or commit, download them as versioned artifacts, and avoid piping remote content directly into bash or sh.",
                        "line_number": base_line + offset,
                        "line_content": line[:240],
                        "file_path": file_path,
                        "rule_id": "PIPE-SCRIPT-001",
                    }
                )
                continue

            eval_match = self._EVAL.search(line)
            if eval_match:
                dynamic_expr = eval_match.group(1)
                is_untrusted = bool(re.search(r"(?i)(untrusted|github\.event|pull_request|\$\{|\$[A-Z_][A-Z0-9_]*)", dynamic_expr))
                findings.append(
                    {
                        "type": "script_injection",
                        "confidence": "high" if is_untrusted else "medium",
                        "severity": "HIGH" if is_untrusted else "MEDIUM",
                        "issue": "Dynamic command execution via eval",
                        "explanation": "The pipeline uses eval to execute dynamically constructed shell code. When the command includes untrusted input, it becomes a direct command injection path.",
                        "fix": "Remove eval, pass arguments explicitly, and validate or sanitize all user-controlled variables before shell execution.",
                        "line_number": base_line + offset,
                        "line_content": line[:240],
                        "file_path": file_path,
                        "rule_id": "PIPE-SCRIPT-002",
                    }
                )
                continue

            if self._EXEC.search(line):
                findings.append(
                    {
                        "type": "script_injection",
                        "confidence": "medium",
                        "severity": "MEDIUM",
                        "issue": "Dynamic exec usage in pipeline script",
                        "explanation": "The script uses exec-style execution, which is risky when command content can be influenced by external inputs or repository content.",
                        "fix": "Prefer explicit command invocations and ensure command content is not dynamically assembled from untrusted data.",
                        "line_number": base_line + offset,
                        "line_content": line[:240],
                        "file_path": file_path,
                        "rule_id": "PIPE-SCRIPT-003",
                    }
                )
        return findings