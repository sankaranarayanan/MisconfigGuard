"""
EntropyAnalyzer — Detect high-randomness strings that may be secrets.

Uses Shannon entropy to identify strings that are likely cryptographic
keys, tokens, or other sensitive random values embedded in source code.
"""

from __future__ import annotations

import math
import re
from typing import List, Tuple

# Patterns for candidate high-entropy tokens.
# B64_PATTERN matches base64-like characters (common for API keys, tokens, certs).
# HEX_PATTERN matches long hex strings (common for hashes, GUIDs used as secrets).
_B64_PATTERN = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
_HEX_PATTERN = re.compile(r"[0-9a-fA-F]{32,}")


def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string (bits per character)."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
    )


class EntropyAnalyzer:
    """
    Detects potentially secret strings by measuring Shannon entropy.

    High-entropy long strings are likely cryptographic keys or tokens.
    Strings below the entropy threshold (e.g. repeated chars, short values)
    are ignored to keep the false-positive rate low.
    """

    def __init__(
        self,
        entropy_threshold: float = 4.5,
        min_length: int = 20,
    ) -> None:
        self._entropy_threshold = entropy_threshold
        self._min_length = min_length

    def analyze_string(self, s: str) -> float:
        """Return the Shannon entropy of the given string."""
        return _shannon_entropy(s)

    def is_high_entropy(self, s: str) -> bool:
        """True when the string meets both the length and entropy thresholds."""
        return (
            len(s) >= self._min_length
            and _shannon_entropy(s) >= self._entropy_threshold
        )

    def scan_text(self, text: str) -> List[Tuple[str, float]]:
        """
        Scan text for high-entropy tokens.

        Returns a list of (token, entropy) pairs for tokens that exceed
        both the length and entropy thresholds.
        """
        findings: List[Tuple[str, float]] = []
        seen: set[str] = set()

        for pattern in (_B64_PATTERN, _HEX_PATTERN):
            for match in pattern.finditer(text):
                token = match.group()
                if token in seen:
                    continue
                entropy = _shannon_entropy(token)
                if len(token) >= self._min_length and entropy >= self._entropy_threshold:
                    findings.append((token, entropy))
                    seen.add(token)

        return findings
