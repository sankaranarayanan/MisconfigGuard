"""
LLMGuardrails — validate and enforce structure on LLM outputs.

Protects against:
    • Hallucinated / out-of-schema fields in structured responses
    • Severity values outside the allowed set
    • Missing mandatory fields
    • Outputs that re-echo prompt-injection patterns back to the caller
    • Excessively long single fields (DoS / junk output)
    • Outputs that reference disallowed external URLs

Provides:
    OutputValidationError — raised when output fails a hard check
    LLMGuardrails         — main guard class

Usage
-----
    guardrails = LLMGuardrails()

    # Validate a structured issue dict
    clean_issue = guardrails.validate_issue(raw_issue)

    # Validate a full analysis result (list of issues)
    clean_result = guardrails.validate_result(raw_result)

    # Enforce output schema on an arbitrary dict
    enforced = guardrails.enforce_schema(raw_dict, schema=MY_SCHEMA)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set


class OutputValidationError(ValueError):
    """Raised when an LLM output fails schema or content validation."""


# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------

_ALLOWED_SEVERITIES: Set[str] = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO", "UNKNOWN"}

# Standard CWE format
_CWE_RE = re.compile(r"^CWE-\d+$", re.IGNORECASE)

# Repeated injection patterns that should never appear in output
_OUTPUT_INJECTION_RE = re.compile(
    r"(?:ignore|forget|override)\s+(?:previous|prior|all)\s+(?:instructions?|rules?|context)",
    re.IGNORECASE,
)

# External URL pattern for disallowed domains (data-exfiltration via LLM output)
_SUSPICIOUS_URL_RE = re.compile(
    r"https?://(?!(?:nvd\.nist\.gov|owasp\.org|cve\.mitre\.org|docs\.microsoft\.com|cloud\.google\.com|docs\.aws\.amazon\.com))[^\s]{10,}",
    re.IGNORECASE,
)

_MAX_FIELD_LENGTH = 8_000   # characters; beyond this a single field is suspicious

# ---------------------------------------------------------------------------
# Issue schema
# ---------------------------------------------------------------------------

# Mandatory keys every issue must contain
_ISSUE_MANDATORY_KEYS = {"title", "severity", "description"}

# All keys allowed in a structured issue dict
_ISSUE_ALLOWED_KEYS = {
    "title", "severity", "description",
    "affected_resource", "recommendation",
    "cwe", "owasp", "rule_id", "rule_description",
    "file_path", "line_number",
}

# ---------------------------------------------------------------------------
# LLMGuardrails
# ---------------------------------------------------------------------------

class LLMGuardrails:
    """
    Validates and enforces schema on LLM-generated outputs.

    Parameters
    ----------
    strip_unknown_fields:
        If True (default), unknown fields are silently dropped rather than
        raising an error.
    allow_external_urls:
        If True, external URLs in output are not flagged.
    max_field_length:
        Maximum allowed length for any single string field.
    """

    def __init__(
        self,
        strip_unknown_fields: bool = True,
        allow_external_urls: bool = False,
        max_field_length: int = _MAX_FIELD_LENGTH,
    ) -> None:
        self.strip_unknown_fields = strip_unknown_fields
        self.allow_external_urls = allow_external_urls
        self.max_field_length = max_field_length

    # ------------------------------------------------------------------
    # Single-issue validation
    # ------------------------------------------------------------------

    def validate_issue(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean a single structured issue dict.

        Raises ``OutputValidationError`` on hard failures.
        Returns a cleaned copy.
        """
        if not isinstance(issue, dict):
            raise OutputValidationError(f"Issue must be a dict, got {type(issue).__name__}")

        result: Dict[str, Any] = {}

        # 1. Strip or reject unknown fields
        for key, value in issue.items():
            if key not in _ISSUE_ALLOWED_KEYS:
                if self.strip_unknown_fields:
                    continue   # silently drop
                raise OutputValidationError(f"Unknown field in issue: '{key}'")
            result[key] = value

        # 2. Mandatory fields
        for key in _ISSUE_MANDATORY_KEYS:
            if not result.get(key):
                raise OutputValidationError(f"Issue missing mandatory field: '{key}'")

        # 3. Severity normalisation and validation
        raw_sev = str(result.get("severity", "")).strip().upper()
        if raw_sev not in _ALLOWED_SEVERITIES:
            # Attempt fuzzy fix
            raw_sev = self._normalise_severity(raw_sev)
        result["severity"] = raw_sev

        # 4. CWE format check
        cwe = result.get("cwe", "")
        if cwe and not _CWE_RE.match(str(cwe)):
            result["cwe"] = ""   # clear malformed CWE rather than propagating

        # 5.Content safety checks on all string fields
        for key, value in result.items():
            if not isinstance(value, str):
                continue
            self._check_field_safety(key, value)

        return result

    def validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a full analysis result dict (containing an ``issues`` list).

        Returns a cleaned copy.
        """
        if not isinstance(result, dict):
            raise OutputValidationError(f"Result must be a dict, got {type(result).__name__}")

        clean = dict(result)
        raw_issues = result.get("issues", [])
        if not isinstance(raw_issues, list):
            raise OutputValidationError("'issues' field must be a list")

        clean_issues = []
        for i, issue in enumerate(raw_issues):
            try:
                clean_issues.append(self.validate_issue(issue))
            except OutputValidationError as exc:
                # Log and skip invalid issues rather than failing the whole result
                import logging
                logging.getLogger(__name__).warning(
                    "Dropping invalid issue at index %d: %s", i, exc
                )
        clean["issues"] = clean_issues
        return clean

    # ------------------------------------------------------------------
    # Generic schema enforcement
    # ------------------------------------------------------------------

    def enforce_schema(
        self,
        data: Dict[str, Any],
        *,
        schema: Dict[str, type],
        required: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Enforce a type schema on an arbitrary dict.

        Parameters
        ----------
        data:
            The dict to validate.
        schema:
            Mapping of field name → expected Python type.
        required:
            List of fields that must be present and non-empty.

        Returns a cleaned copy with type-coerced values.
        Raises ``OutputValidationError`` for unrecoverable violations.
        """
        if not isinstance(data, dict):
            raise OutputValidationError(f"Expected dict, got {type(data).__name__}")

        clean: Dict[str, Any] = {}
        for key, expected_type in schema.items():
            value = data.get(key)
            if value is None:
                clean[key] = None
                continue
            if not isinstance(value, expected_type):
                try:
                    value = expected_type(value)
                except (TypeError, ValueError) as exc:
                    raise OutputValidationError(
                        f"Field '{key}' cannot be coerced to {expected_type.__name__}: {exc}"
                    ) from exc
            clean[key] = value

        # Keep any extra keys not in schema (non-strict by default)
        for key, value in data.items():
            if key not in clean:
                clean[key] = value

        # Required field check
        for key in (required or []):
            if not clean.get(key):
                raise OutputValidationError(f"Required field missing or empty: '{key}'")

        return clean

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_field_safety(self, field_name: str, value: str) -> None:
        """Check a string field for injection and length issues."""
        if len(value) > self.max_field_length:
            raise OutputValidationError(
                f"Field '{field_name}' exceeds maximum allowed length "
                f"({len(value)} > {self.max_field_length})"
            )
        if _OUTPUT_INJECTION_RE.search(value):
            raise OutputValidationError(
                f"Field '{field_name}' contains prompt-injection patterns in LLM output"
            )
        if not self.allow_external_urls and _SUSPICIOUS_URL_RE.search(value):
            import logging
            logging.getLogger(__name__).warning(
                "Field '%s' contains suspicious external URL — redacted", field_name
            )
            # Redact rather than hard-fail to avoid blocking legitimate CVE links
            import re as _re
            value = _SUSPICIOUS_URL_RE.sub("[URL REDACTED]", value)

    @staticmethod
    def _normalise_severity(raw: str) -> str:
        """Map non-standard severity strings to the canonical set."""
        _MAP = {
            "CRIT": "CRITICAL",
            "CRITICAL": "CRITICAL",
            "ERROR": "HIGH",
            "WARN": "MEDIUM",
            "WARNING": "MEDIUM",
            "NOTE": "LOW",
            "NOTICE": "LOW",
            "DEBUG": "INFO",
        }
        for key, canonical in _MAP.items():
            if key in raw:
                return canonical
        return "UNKNOWN"
