"""
InputSanitizer — validate and sanitize all external inputs entering the pipeline.

Protects against:
    • Overly long inputs (DoS / memory exhaustion)
    • Null-byte injection
    • Path traversal sequences
    • HTML / script tag injection
    • Prompt injection primitives (role-override markers, jailbreak patterns)
    • Control characters

Usage
-----
    sanitizer = InputSanitizer()
    clean = sanitizer.sanitize(user_text)          # raises SanitizationError on violation
    ok, reason = sanitizer.validate(user_text)     # non-raising check

    # Custom limits
    sanitizer = InputSanitizer(max_length=512, allow_newlines=False)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


class SanitizationError(ValueError):
    """Raised when an input fails sanitization checks."""


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# HTML / script injection
_HTML_TAG_RE = re.compile(r"<[a-zA-Z/!?][^>]{0,200}>", re.DOTALL)

# Path traversal (both POSIX and Windows style)
_PATH_TRAVERSAL_RE = re.compile(r"\.\.[/\\]")

# Prompt-injection primitives  ─  attempts to override system instructions
_PROMPT_INJECTION_RE = re.compile(
    r"""
    (?:ignore|forget|disregard|override|bypass)\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|context|rules?)
    | system\s*:\s*you\s+are
    | <\s*/?system\s*>
    | \[\s*system\s*\]
    | (INST|SYS)\s*>>
    | <<\s*(INST|SYS)
    | \|\s*im_start\s*\|
    | \|\s*im_end\s*\|
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Null bytes and non-printable control characters (keep tab, LF, CR)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class InputSanitizer:
    """
    Configurable input sanitizer.

    Parameters
    ----------
    max_length:
        Maximum allowed length in characters (default 32 768).
    allow_html:
        If False (default) any HTML-like tags cause rejection.
    allow_path_traversal:
        If False (default) ``../`` or ``..\\`` sequences cause rejection.
    allow_prompt_injection_patterns:
        If False (default) known jailbreak / override patterns cause rejection.
    strip_control_chars:
        Strip (rather than reject) non-printable control characters (default True).
    allow_newlines:
        If True (default) ``\\n`` and ``\\r`` pass through unchanged.
    extra_blocked_patterns:
        Additional compiled regex patterns that should trigger rejection.
    """

    max_length: int = 32_768
    allow_html: bool = False
    allow_path_traversal: bool = False
    allow_prompt_injection_patterns: bool = False
    strip_control_chars: bool = True
    allow_newlines: bool = True
    extra_blocked_patterns: List[re.Pattern] = field(default_factory=list)

    def sanitize(self, text: str, *, context: str = "input") -> str:
        """
        Return a sanitized copy of *text*.

        Raises
        ------
        SanitizationError
            If the text violates any hard limit (length, injections, …).
        TypeError
            If *text* is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        # Normalise unicode to NFC to avoid homoglyph bypasses
        text = unicodedata.normalize("NFC", text)

        # 1. Strip or reject control characters
        if self.strip_control_chars:
            text = _CONTROL_CHAR_RE.sub("", text)
        elif _CONTROL_CHAR_RE.search(text):
            raise SanitizationError(
                f"[{context}] Input contains non-printable control characters"
            )

        # 2. Null-byte check (belt-and-suspenders after strip)
        if "\x00" in text:
            raise SanitizationError(f"[{context}] Input contains null bytes")

        # 3. Length limit
        if len(text) > self.max_length:
            raise SanitizationError(
                f"[{context}] Input exceeds maximum length "
                f"({len(text)} > {self.max_length})"
            )

        # 4. HTML tags
        if not self.allow_html and _HTML_TAG_RE.search(text):
            raise SanitizationError(
                f"[{context}] Input contains disallowed HTML/markup tags"
            )

        # 5. Path traversal
        if not self.allow_path_traversal and _PATH_TRAVERSAL_RE.search(text):
            raise SanitizationError(
                f"[{context}] Input contains path-traversal sequences"
            )

        # 6. Prompt injection
        if not self.allow_prompt_injection_patterns and _PROMPT_INJECTION_RE.search(text):
            raise SanitizationError(
                f"[{context}] Input contains prompt-injection patterns"
            )

        # 7. Extra blocked patterns
        for pat in self.extra_blocked_patterns:
            if pat.search(text):
                raise SanitizationError(
                    f"[{context}] Input matches blocked pattern: {pat.pattern!r}"
                )

        # 8. Newlines
        if not self.allow_newlines:
            text = text.replace("\n", " ").replace("\r", " ")

        return text

    def validate(self, text: str, *, context: str = "input") -> Tuple[bool, Optional[str]]:
        """
        Non-raising variant.  Returns ``(True, None)`` on success or
        ``(False, reason)`` on failure.
        """
        try:
            self.sanitize(text, context=context)
            return True, None
        except (SanitizationError, TypeError) as exc:
            return False, str(exc)

    def sanitize_dict(self, data: dict, *, context: str = "input") -> dict:
        """
        Recursively sanitize all string values in a dictionary.
        Non-string leaf values are returned unchanged.
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.sanitize(value, context=f"{context}.{key}")
            elif isinstance(value, dict):
                result[key] = self.sanitize_dict(value, context=f"{context}.{key}")
            elif isinstance(value, list):
                result[key] = self._sanitize_list(value, context=f"{context}.{key}")
            else:
                result[key] = value
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sanitize_list(self, items: list, *, context: str) -> list:
        result = []
        for i, item in enumerate(items):
            if isinstance(item, str):
                result.append(self.sanitize(item, context=f"{context}[{i}]"))
            elif isinstance(item, dict):
                result.append(self.sanitize_dict(item, context=f"{context}[{i}]"))
            elif isinstance(item, list):
                result.append(self._sanitize_list(item, context=f"{context}[{i}]"))
            else:
                result.append(item)
        return result
