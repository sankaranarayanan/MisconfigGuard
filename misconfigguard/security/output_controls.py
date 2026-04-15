"""
OutputControls — filter, redact, and control what leaves the pipeline.

Responsibilities:
    • Redact secrets / credentials that may have leaked into output
    • Enforce severity-based output filtering (only surface what the caller
      is allowed to see according to their role)
    • Cap the number of issues surfaced per report
    • Strip internal-only metadata fields before external delivery
    • Mark output as ``sanitized`` so downstream consumers know it has
      passed through the output control layer

Usage
-----
    controls = OutputControls()

    # Role-filtered report
    filtered = controls.filter_for_role(result, role="viewer")

    # Redact secrets from any dict
    clean = controls.redact_secrets(result)

    # Full output pipeline
    ready = controls.process(result, role="analyst", max_issues=50)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Redaction patterns
# ---------------------------------------------------------------------------

# Common secret / credential patterns
_SECRET_PATTERNS: List[re.Pattern] = [
    # Generic API keys / tokens
    re.compile(r"(?i)(?:api[_\-]?key|apikey|access[_\-]?token|auth[_\-]?token|bearer)\s*[:=]\s*['\"]?([A-Za-z0-9\-_]{16,64})['\"]?"),
    # AWS credentials
    re.compile(r"(?i)AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)aws[_\-]?secret[_\-]?access[_\-]?key\s*[:=]\s*['\"]?([A-Za-z0-9+/]{40})['\"]?"),
    # GitHub tokens
    re.compile(r"(?i)ghp_[A-Za-z0-9]{36}"),
    re.compile(r"(?i)github[_\-]?token\s*[:=]\s*['\"]?([A-Za-z0-9\-_]{20,})['\"]?"),
    # Generic passwords
    re.compile(r"(?i)password\s*[:=]\s*['\"]?([^\s,'\"]{8,})['\"]?"),
    # PEM private key
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    # Connection strings
    re.compile(r"(?i)(?:connection[_\-]?string|conn[_\-]?str)\s*[:=]\s*['\"]?([^'\"]{10,})['\"]?"),
]

_REDACTION_PLACEHOLDER = "[REDACTED]"

# ---------------------------------------------------------------------------
# Severity ordering (higher index = lower priority)
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO", "UNKNOWN"]

# Role → minimum severity the role can see
_ROLE_MIN_SEVERITY: Dict[str, str] = {
    "viewer":   "HIGH",      # viewers only see HIGH and above
    "analyst":  "MEDIUM",
    "engineer": "LOW",
    "admin":    "INFO",
}

# Internal metadata fields stripped before external delivery
_INTERNAL_FIELDS: Set[str] = {
    "_validation", "trust_score", "trust_source", "_cache_key", "_embedding",
}


class OutputControls:
    """
    Post-processing filter applied to all pipeline outputs before delivery.

    Parameters
    ----------
    redact_secrets:
        If True (default), apply secret-pattern redaction to string fields.
    strip_internal_metadata:
        If True (default), remove internal-only metadata keys.
    default_max_issues:
        Default cap on the number of issues in the output (0 = unlimited).
    """

    def __init__(
        self,
        redact_secrets: bool = True,
        strip_internal_metadata: bool = True,
        default_max_issues: int = 0,
    ) -> None:
        self._do_redact = redact_secrets
        self._strip_internal = strip_internal_metadata
        self.default_max_issues = default_max_issues

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        result: Dict[str, Any],
        *,
        role: str = "analyst",
        max_issues: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the full output control pipeline:
        1. Redact secrets
        2. Filter by role severity
        3. Cap issue count
        4. Strip internal metadata
        5. Mark as sanitized

        Returns a cleaned copy.
        """
        out = dict(result)

        if self._do_redact:
            out = self.redact_secrets(out)

        out = self.filter_for_role(out, role=role)

        cap = max_issues if max_issues is not None else self.default_max_issues
        if cap > 0:
            issues = out.get("issues", [])
            if isinstance(issues, list) and len(issues) > cap:
                out["issues"] = issues[:cap]
                out.setdefault("metadata", {})["truncated"] = True

        if self._strip_internal:
            out = self._strip_internal_metadata(out)

        out.setdefault("metadata", {})
        out["metadata"]["output_sanitized"] = True

        # Rebuild summary counts after filtering
        issues_list = out.get("issues", [])
        if isinstance(issues_list, list):
            out["summary"] = self._recount_summary(issues_list)

        return out

    def filter_for_role(self, result: Dict[str, Any], *, role: str) -> Dict[str, Any]:
        """
        Return a copy of *result* with issues filtered to those the *role*
        is permitted to see, based on minimum severity.
        """
        min_sev = _ROLE_MIN_SEVERITY.get(role.lower(), "INFO")
        min_idx = _SEVERITY_ORDER.index(min_sev) if min_sev in _SEVERITY_ORDER else 0

        out = dict(result)
        issues = out.get("issues", [])
        if not isinstance(issues, list):
            return out

        def _passes(issue: Dict[str, Any]) -> bool:
            sev = str(issue.get("severity", "UNKNOWN")).upper()
            try:
                return _SEVERITY_ORDER.index(sev) <= min_idx
            except ValueError:
                return True  # unknown severity: pass through

        out["issues"] = [i for i in issues if _passes(i)]
        return out

    def redact_secrets(self, obj: Any) -> Any:
        """
        Recursively redact secret patterns from all string values in *obj*.
        Works on dicts, lists, and plain strings.
        """
        if isinstance(obj, str):
            return self._redact_string(obj)
        if isinstance(obj, dict):
            return {k: self.redact_secrets(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.redact_secrets(item) for item in obj]
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _redact_string(self, text: str) -> str:
        for pattern in _SECRET_PATTERNS:
            text = pattern.sub(lambda m: m.group(0)[:max(0, m.start(1) - m.start())] + _REDACTION_PLACEHOLDER if m.lastindex else _REDACTION_PLACEHOLDER, text)
        return text

    def _strip_internal_metadata(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: self._strip_internal_metadata(v)
                for k, v in obj.items()
                if k not in _INTERNAL_FIELDS
            }
        if isinstance(obj, list):
            return [self._strip_internal_metadata(item) for item in obj]
        return obj

    @staticmethod
    def _recount_summary(issues: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for issue in issues:
            sev = str(issue.get("severity", "")).lower()
            if sev in counts:
                counts[sev] += 1
        return counts
