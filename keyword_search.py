"""
KeywordSearchEngine — BM25-based lexical search over chunk text.

Complements semantic (vector) search by capturing exact security-critical
terms that embedding models may under-weight: "0.0.0.0/0", "public-read",
"AllowAll", "wildcard", hardcoded secrets, etc.

Backend selection (automatic, best available)
---------------------------------------------
1. rank_bm25  — BM25Okapi  (pip install rank-bm25)   ← preferred
2. sklearn    — TF-IDF + cosine (pip install scikit-learn)
3. built-in   — Token overlap with IDF weighting (zero extra deps)

All backends expose the same interface and return min-max normalised scores
in [0, 1] so they can be linearly combined with semantic similarity scores.

Usage
-----
    engine = KeywordSearchEngine()
    engine.index(chunks)                        # build index
    results = engine.search("public S3 bucket", top_k=5)
    # → [(chunk_dict, normalised_score), …]
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional backend imports
# ---------------------------------------------------------------------------

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

try:
    import numpy as _np
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared tokeniser
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """
    Split text into searchable tokens.

    Splits on whitespace, underscores, hyphens, and other non-alphanumeric
    characters (preserving dots and slashes for IP addresses like ``0.0.0.0/0``).
    For example, ``aws_s3_bucket`` → ``["aws", "s3", "bucket"]``,
    ``public-read`` → ``["public", "read"]``.
    Returns lower-cased tokens with at least 2 characters.
    """
    raw = re.split(r"[^a-zA-Z0-9./]+", text.lower())
    return [t for t in raw if len(t) >= 2]


# ---------------------------------------------------------------------------
# KeywordSearchEngine
# ---------------------------------------------------------------------------


class KeywordSearchEngine:
    """
    Lexical keyword search over a collection of text chunks.

    Scoring is min-max normalised so results can be blended with
    vector-similarity scores (0 = no match, 1 = best match in corpus).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """
        Args:
            k1: BM25 term-saturation parameter (only used with rank_bm25).
            b:  BM25 length-normalisation parameter (only used with rank_bm25).
        """
        self.k1 = k1
        self.b  = b

        self._chunks: List[dict] = []
        self._backend = self._select_backend()

        # Backend-specific state
        self._bm25        = None   # rank_bm25 model
        self._tfidf_vec   = None   # sklearn TfidfVectorizer
        self._tfidf_mat   = None   # sklearn document-term matrix
        self._tokenized   : List[List[str]] = []   # for bm25 / built-in
        self._idf         : Dict[str, float] = {}  # built-in IDF weights

        logger.debug("KeywordSearchEngine backend: %s", self._backend)

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_backend() -> str:
        if _BM25_AVAILABLE:
            return "bm25"
        if _SKLEARN_AVAILABLE:
            return "tfidf"
        return "builtin"

    @property
    def backend(self) -> str:
        """Name of the active search backend."""
        return self._backend

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, chunks: List[dict]) -> None:
        """
        Build (or rebuild) the keyword index from *chunks*.

        Each chunk must have a ``"text"`` field.  Calling ``index()``
        replaces the existing index entirely.
        """
        self._chunks = list(chunks)
        texts = [c.get("text", "") for c in self._chunks]

        if self._backend == "bm25":
            tokenized = [_tokenize(t) for t in texts]
            self._tokenized = tokenized
            if not tokenized:
                self._bm25 = None  # can't build BM25 on empty corpus
            else:
                self._bm25 = _BM25Okapi(tokenized, k1=self.k1, b=self.b)

        elif self._backend == "tfidf":
            self._tfidf_vec = _TfidfVectorizer(
                token_pattern=r"[a-zA-Z0-9_./-]{2,}",
                lowercase=True,
                sublinear_tf=True,
            )
            if texts:
                self._tfidf_mat = self._tfidf_vec.fit_transform(texts)

        else:  # built-in
            self._tokenized = [_tokenize(t) for t in texts]
            self._build_idf()

        logger.debug(
            "KeywordSearchEngine indexed %d chunks (backend=%s)",
            len(self._chunks), self._backend,
        )

    def _build_idf(self) -> None:
        """Compute IDF weights for the built-in backend."""
        N = len(self._tokenized)
        df: Counter = Counter()
        for tokens in self._tokenized:
            df.update(set(tokens))
        self._idf = {
            tok: math.log((N + 1) / (freq + 1)) + 1.0
            for tok, freq in df.items()
        }

    def add_chunks(self, chunks: List[dict]) -> None:
        """Append *chunks* to the existing index and rebuild."""
        self.index(self._chunks + chunks)

    def clear(self) -> None:
        """Reset the engine to an empty state."""
        self._chunks = []
        self._bm25 = None
        self._tfidf_vec = None
        self._tfidf_mat = None
        self._tokenized = []
        self._idf = {}

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[dict, float]]:
        """
        Return up to *top_k* ``(chunk_dict, normalised_score)`` pairs,
        sorted by descending keyword relevance.

        Scores are min-max normalised to ``[0, 1]``.
        Zero-score chunks are excluded from the results.

        Args:
            query:  The search query string.
            top_k:  Maximum number of results to return.

        Returns:
            List of (chunk_dict, score) tuples.
        """
        if not self._chunks:
            return []
        if self._backend == "bm25":
            return self._search_bm25(query, top_k)
        if self._backend == "tfidf":
            return self._search_tfidf(query, top_k)
        return self._search_builtin(query, top_k)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _minmax(scores: List[float]) -> List[float]:
        """Min-max normalise *scores* to [0, 1]."""
        if not scores:
            return scores
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return [1.0 if s > 0 else 0.0 for s in scores]
        span = hi - lo
        return [(s - lo) / span for s in scores]

    def _search_bm25(self, query: str, top_k: int) -> List[Tuple[dict, float]]:
        if self._bm25 is None:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        raw = list(self._bm25.get_scores(q_tokens))
        normed = self._minmax(raw)
        ranked = sorted(
            zip(normed, self._chunks),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            (chunk, score) for score, chunk in ranked[:top_k] if score > 0
        ]

    def _search_tfidf(self, query: str, top_k: int) -> List[Tuple[dict, float]]:
        if self._tfidf_vec is None or self._tfidf_mat is None:
            return []
        q_vec = self._tfidf_vec.transform([query])
        raw   = _cosine_sim(q_vec, self._tfidf_mat).flatten().tolist()
        normed = self._minmax(raw)
        ranked = sorted(enumerate(normed), key=lambda x: x[1], reverse=True)
        return [
            (self._chunks[i], s) for i, s in ranked[:top_k] if s > 0
        ]

    def _search_builtin(self, query: str, top_k: int) -> List[Tuple[dict, float]]:
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []
        raw: List[float] = []
        for tokens in self._tokenized:
            overlap = q_tokens & set(tokens)
            score = sum(self._idf.get(t, 1.0) for t in overlap)
            raw.append(score)
        normed = self._minmax(raw)
        ranked = sorted(enumerate(normed), key=lambda x: x[1], reverse=True)
        return [
            (self._chunks[i], s) for i, s in ranked[:top_k] if s > 0
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_indexed(self) -> int:
        """Number of chunks currently in the keyword index."""
        return len(self._chunks)
