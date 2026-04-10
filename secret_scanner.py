"""
SecretScanner — Regex and entropy-based detection of hardcoded secrets.

Scans text content from Terraform, YAML, and JSON files for common secret
patterns, including AWS keys, Azure connection strings, passwords, API keys,
tokens, and private key headers. High-entropy strings are also flagged via
the EntropyAnalyzer as low-confidence candidates.

Matched values are ALWAYS masked before storage — the raw secret is never
retained so that test output and logs cannot leak credentials.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from entropy_analyzer import EntropyAnalyzer

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"


# ---------------------------------------------------------------------------
# SecretMatch dataclass
# ---------------------------------------------------------------------------

@dataclass
class SecretMatch:
    """A single detected (and masked) potential secret."""

    match: str          # Masked value — NEVER the raw secret
    secret_type: str    # e.g. "aws_key", "password", "api_key", "token"
    confidence: str     # "high" | "medium" | "low"
    severity: str       # "HIGH" | "MEDIUM" | "LOW"
    line_number: int
    line_content: str   # Context line with the secret value masked
    file_path: str = ""

    def to_dict(self) -> dict:
        return {
            "match": self.match,
            "secret_type": self.secret_type,
            "confidence": self.confidence,
            "severity": self.severity,
            "line_number": self.line_number,
            "line_content": self.line_content,
            "file_path": self.file_path,
        }


# ---------------------------------------------------------------------------
# Masking helper
# ---------------------------------------------------------------------------

def _mask_value(value: str) -> str:
    """
    Mask a secret value for safe display.

    Preserves the first 4 and last 2 characters to help identify the
    pattern class while preventing the full value from being logged.
    """
    if len(value) <= 8:
        return "***"
    return value[:4] + "***" + value[-2:]


# ---------------------------------------------------------------------------
# Placeholder filter
# ---------------------------------------------------------------------------

_PLACEHOLDER_EXACT: frozenset[str] = frozenset({
    "changeme", "placeholder", "your_password", "your_api_key",
    "your_token", "replace_me", "enter_your", "put_your", "example",
    "xxxx", "yyyy", "zzzz", "test", "demo", "sample",
    "xxxxxx", "yyyyyy", "todo", "fixme", "admin", "password",
    "secret", "null", "none", "", "true", "false",
})

_PLACEHOLDER_RE = re.compile(
    r"(?:change|replace|your|enter|put|placeholder|example|test|todo|xxxx|fixme)",
    re.IGNORECASE,
)


def _is_placeholder(value: str) -> bool:
    """Return True if the matched value appears to be a placeholder, not a real secret."""
    stripped = value.strip().strip('"').strip("'").lower()
    if stripped in _PLACEHOLDER_EXACT:
        return True
    if _PLACEHOLDER_RE.search(stripped):
        return True
    # Very short values are not real secrets
    if len(stripped) < 4:
        return True
    return False


# ---------------------------------------------------------------------------
# Compiled regex pattern registry
# Each entry: (compiled_regex, secret_type, confidence, severity)
# ---------------------------------------------------------------------------

_PATTERNS: List[Tuple[re.Pattern, str, str, str]] = [
    # AWS Access Key ID (always starts with AKIA)
    (
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "aws_key", "high", HIGH,
    ),
    # AWS Secret Access Key (longer, follows "secret" context)
    (
        re.compile(
            r"(?i)aws.{0,40}secret.{0,20}key\s*[=:]\s*[\"']?([A-Za-z0-9/+]{40})[\"']?"
        ),
        "aws_secret", "high", HIGH,
    ),
    # Azure storage account key (AccountKey= followed by base64)
    (
        re.compile(r"AccountKey=[A-Za-z0-9+/]{40,}={0,2}"),
        "azure_storage_key", "high", HIGH,
    ),
    # Azure connection string starting marker (MEDIUM — may contain embedded key)
    (
        re.compile(r"DefaultEndpointsProtocol=[a-z]+;"),
        "azure_connection_string", "high", MEDIUM,
    ),
    # Private key PEM header (any standard key type)
    (
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
        ),
        "private_key", "high", HIGH,
    ),
    # Certificate PEM header
    (
        re.compile(r"-----BEGIN CERTIFICATE-----"),
        "certificate", "medium", MEDIUM,
    ),
    # Password/passwd/pwd assignment
    (
        re.compile(
            r"(?i)(?:password|passwd|pwd)[\"']?\s*[=:]\s*[\"']?([^\s\"'#]{4,})[\"']?"
        ),
        "password", "high", HIGH,
    ),
    # API key assignment (api_key, api-key, apikey)
    (
        re.compile(
            # Allow an optional closing quote after the key name to handle
            # JSON format:  "api_key": "value"
            r"(?i)api[_\-]?key[\"']?\s*[=:]\s*[\"']?([^\s\"'#]{8,})[\"']?"
        ),
        "api_key", "high", HIGH,
    ),
    # Token assignment
    (
        re.compile(
            r"(?i)\btoken[\"']?\s*[=:]\s*[\"']?([^\s\"'#]{8,})[\"']?"
        ),
        "token", "high", HIGH,
    ),
    # Generic "secret" assignment
    (
        re.compile(
            r"(?i)\bsecret[\"']?\s*[=:]\s*[\"']?([^\s\"'#]{4,})[\"']?"
        ),
        "secret", "high", HIGH,
    ),
    # Generic connection string
    (
        re.compile(
            r"(?i)connection[_\-]?string[\"']?\s*[=:]\s*[\"']?([^\s\"'#]{8,})[\"']?"
        ),
        "connection_string", "medium", MEDIUM,
    ),
    # JWT token (three base64url segments separated by dots)
    (
        re.compile(
            r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"
        ),
        "jwt_token", "high", HIGH,
    ),
]


# ---------------------------------------------------------------------------
# SecretScanner
# ---------------------------------------------------------------------------

class SecretScanner:
    """
    Scans text content for hardcoded secrets using regex patterns and
    entropy analysis.

    Matched values are masked before being stored in SecretMatch objects.
    Placeholder/example values are suppressed to keep false positives low.
    """

    def __init__(
        self,
        entropy_threshold: float = 4.5,
        min_entropy_length: int = 20,
    ) -> None:
        self._entropy_analyzer = EntropyAnalyzer(
            entropy_threshold=entropy_threshold,
            min_length=min_entropy_length,
        )

    def scan_text(self, text: str, file_path: str = "") -> List[SecretMatch]:
        """
        Scan text content for potential secrets.

        Returns a list of SecretMatch objects, each with the matched value
        masked. A dedup key prevents the same (line, type, value) appearing
        more than once.
        """
        findings: List[SecretMatch] = []
        seen: set[tuple] = set()
        raw_candidates: set[str] = set()
        lines = text.splitlines()

        for line_num, line in enumerate(lines, 1):
            for pattern, secret_type, confidence, severity in _PATTERNS:
                for m in pattern.finditer(line):
                    raw_match = m.group(0)
                    # Use capture group 1 if present (the extracted value),
                    # otherwise use the full match (e.g. bare patterns like AKIA…)
                    # For bare-match patterns (no capture group), the whole match
                    # IS the secret pattern (e.g., AKIA key, key header) — skip
                    # placeholder filtering as these patterns can't be placeholders.
                    if m.lastindex:
                        value = m.group(1)
                        if _is_placeholder(value):
                            continue
                    else:
                        value = m.group(0)

                    key = (line_num, secret_type, value[:20])
                    if key in seen:
                        continue
                    seen.add(key)
                    raw_candidates.add(raw_match)
                    raw_candidates.add(value)

                    masked_value = _mask_value(value)
                    # Mask the real value in the context line before storing
                    masked_line = (
                        line.replace(value, _mask_value(value))
                        if value in line
                        else line
                    )[:200]

                    findings.append(SecretMatch(
                        match=masked_value,
                        secret_type=secret_type,
                        confidence=confidence,
                        severity=severity,
                        line_number=line_num,
                        line_content=masked_line,
                        file_path=file_path,
                    ))

        # Entropy-based detection for tokens not caught by named patterns
        for token, entropy in self._entropy_analyzer.scan_text(text):
            # Skip tokens already flagged by a named pattern
            already_flagged = any(token in candidate or candidate in token for candidate in raw_candidates)
            if already_flagged or _is_placeholder(token):
                continue

            # Locate the token's line
            for line_num, line in enumerate(lines, 1):
                if token in line:
                    key = (line_num, "high_entropy", token[:20])
                    if key not in seen:
                        seen.add(key)
                        masked_line = line.replace(token, _mask_value(token))[:200]
                        findings.append(SecretMatch(
                            match=_mask_value(token),
                            secret_type="high_entropy",
                            confidence="medium",
                            severity=LOW,
                            line_number=line_num,
                            line_content=masked_line,
                            file_path=file_path,
                        ))
                    break

        return findings

    def scan_file(self, file_path: str) -> List[SecretMatch]:
        """
        Scan a single file for secrets.

        Raises FileNotFoundError if the path does not point to a file.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        content = path.read_text(encoding="utf-8", errors="replace")
        return self.scan_text(content, file_path=str(path))
