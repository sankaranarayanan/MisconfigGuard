"""
RAGPoisonDefense — protect the RAG pipeline from data-poisoning attacks.

Three complementary layers:

1. **Content Validation** (pre-indexing gate)
   Reject or quarantine chunks that exhibit adversarial patterns:
   instruction-override injections, excessively high Shannon entropy
   (potential encoded payloads), anomalous repetition, or explicit
   malicious markers.

2. **Trust-Aware Indexing**
   Every chunk is assigned a ``trust_score`` in [0.0, 1.0] based on:
   • Source classification (git-tracked / internal > external URL > unknown)
   • Author metadata presence
   • Content validation pass / partial-pass / fail score

3. **Filter Retrieval**
   At query time, retrieved chunks are re-scored and those below the
   configured ``min_trust_score`` threshold are silently dropped before
   the context is assembled.

Usage
-----
    defense = RAGPoisonDefense()

    # Before indexing a chunk
    validated = defense.validate_chunk(chunk)          # raises ContentValidationError on critical

    # Attach trust score to chunk metadata
    scored = defense.score_chunk(chunk, source="git")

    # Filter a retrieval result list
    safe_chunks = defense.filter_retrieval(chunks, min_trust_score=0.5)
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class ContentValidationError(ValueError):
    """Raised when a chunk fails critical content validation."""


# ---------------------------------------------------------------------------
# Compiled adversarial patterns
# ---------------------------------------------------------------------------

# Instruction-override attempts embedded in retrieved content
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"(?:ignore|forget|disregard|override|bypass)\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?|context)",
        re.IGNORECASE,
    ),
    re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
    re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
    re.compile(r"\|\s*im_(?:start|end)\s*\|", re.IGNORECASE),
    re.compile(r"<<\s*(?:INST|SYS)\s*>>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]", re.IGNORECASE),
    # "DAN" / jailbreak keywords
    re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
]

# Data-exfiltration / exfil trigger patterns
_EXFIL_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?:curl|wget|fetch|http\.get)\s+['\"]?https?://[^\s'\"]{10,}", re.IGNORECASE),
    re.compile(r"base64\s*\(", re.IGNORECASE),
    re.compile(r"eval\s*\(", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Trust source tiers
# ---------------------------------------------------------------------------

_SOURCE_TRUST: Dict[str, float] = {
    "git":       0.95,   # tracked in a Git repository
    "internal":  0.90,   # internal system file
    "file":      0.80,   # local file (not git-tracked)
    "external":  0.50,   # external URL / unknown origin
    "unknown":   0.40,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy (bits per character) of *text*."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _repetition_ratio(text: str) -> float:
    """Return the fraction of bigrams that are repeated."""
    words = text.split()
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    total = len(bigrams)
    unique = len(set(bigrams))
    return 1.0 - (unique / total) if total else 0.0


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

@dataclass
class RAGPoisonDefense:
    """
    Configurable RAG poisoning defense layer.

    Parameters
    ----------
    entropy_threshold:
        Chunks with Shannon entropy above this value are suspect
        (possible encoded payloads).  Default: 5.5 bits/char.
    repetition_threshold:
        Chunks with a repetition ratio above this value are suspect
        (e.g. endlessly looping text designed to fill context).
        Default: 0.6 (60 % repeated bigrams).
    min_trust_score:
        Default minimum trust score used by ``filter_retrieval``.
    reject_on_injection:
        If True, chunks matching injection patterns raise
        ``ContentValidationError``; otherwise they are only penalised.
    """

    entropy_threshold: float = 5.5
    repetition_threshold: float = 0.6
    min_trust_score: float = 0.5
    reject_on_injection: bool = True

    # ------------------------------------------------------------------
    # 1. Content Validation
    # ------------------------------------------------------------------

    def validate_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single chunk before it enters the vector store.

        Returns the chunk unchanged on success.
        Raises ``ContentValidationError`` for critical violations when
        ``reject_on_injection`` is True.
        Attaches ``_validation`` metadata with score and flags.
        """
        content: str = chunk.get("content", "") or chunk.get("text", "") or ""
        flags: List[str] = []
        score = 1.0   # starts at full trust; penalties deducted below

        # --- Injection pattern check ---
        matched_injections = [p.pattern for p in _INJECTION_PATTERNS if p.search(content)]
        if matched_injections:
            if self.reject_on_injection:
                raise ContentValidationError(
                    f"Chunk content matches known injection patterns: {matched_injections[:3]}"
                )
            flags.append("injection_pattern")
            score -= 0.5

        # --- Exfiltration pattern check ---
        matched_exfil = [p.pattern for p in _EXFIL_PATTERNS if p.search(content)]
        if matched_exfil:
            flags.append("exfil_pattern")
            score -= 0.3

        # --- Shannon entropy check ---
        entropy = _shannon_entropy(content)
        if entropy > self.entropy_threshold:
            flags.append(f"high_entropy:{entropy:.2f}")
            score -= 0.2

        # --- Repetition check ---
        rep_ratio = _repetition_ratio(content)
        if rep_ratio > self.repetition_threshold:
            flags.append(f"high_repetition:{rep_ratio:.2f}")
            score -= 0.2

        score = max(0.0, score)

        chunk = dict(chunk)
        chunk.setdefault("metadata", {})
        chunk["metadata"] = {**chunk.get("metadata", {}), "_validation": {"score": round(score, 3), "flags": flags}}
        return chunk

    # ------------------------------------------------------------------
    # 2. Trust-Aware Indexing
    # ------------------------------------------------------------------

    def score_chunk(
        self,
        chunk: Dict[str, Any],
        *,
        source: str = "unknown",
        author: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Assign a ``trust_score`` to *chunk* before indexing.

        Parameters
        ----------
        source:
            One of ``"git"``, ``"internal"``, ``"file"``, ``"external"``,
            ``"unknown"``.
        author:
            If the chunk has known authorship, a small bonus is applied.
        """
        chunk = dict(chunk)
        chunk.setdefault("metadata", {})

        # Base score from source tier
        base = _SOURCE_TRUST.get(source.lower(), _SOURCE_TRUST["unknown"])

        # Bonus for known authorship
        if author:
            base = min(1.0, base + 0.05)

        # Factor in any pre-existing validation score
        validation = chunk["metadata"].get("_validation", {})
        val_score = validation.get("score", 1.0)

        trust = round(base * val_score, 3)
        chunk["metadata"] = {
            **chunk.get("metadata", {}),
            "trust_score": trust,
            "trust_source": source,
        }
        return chunk

    # ------------------------------------------------------------------
    # 3. Filter Retrieval
    # ------------------------------------------------------------------

    def filter_retrieval(
        self,
        chunks: List[Dict[str, Any]],
        *,
        min_trust_score: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split retrieved chunks into (trusted, rejected) lists.

        Parameters
        ----------
        chunks:
            List of chunk dicts as returned by the retriever.
        min_trust_score:
            Override the instance-level threshold.

        Returns
        -------
        (trusted, rejected)
            trusted  — chunks that meet the threshold (safe to use)
            rejected — chunks that failed (for audit purposes)
        """
        threshold = min_trust_score if min_trust_score is not None else self.min_trust_score
        trusted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        for chunk in chunks:
            meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
            trust = meta.get("trust_score", 1.0)   # default full trust if unscored
            if trust >= threshold:
                trusted.append(chunk)
            else:
                rejected.append(chunk)

        return trusted, rejected

    # ------------------------------------------------------------------
    # Convenience: validate + score in one call
    # ------------------------------------------------------------------

    def process_chunk(
        self,
        chunk: Dict[str, Any],
        *,
        source: str = "unknown",
        author: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate then score a chunk.  Raises on critical violations."""
        chunk = self.validate_chunk(chunk)
        chunk = self.score_chunk(chunk, source=source, author=author)
        return chunk

    # ------------------------------------------------------------------
    # Content hash (deduplication + tamper detection)
    # ------------------------------------------------------------------

    @staticmethod
    def content_hash(chunk: Dict[str, Any]) -> str:
        """Return a SHA-256 hex digest of the chunk's content."""
        content = chunk.get("content", "") or chunk.get("text", "") or ""
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
