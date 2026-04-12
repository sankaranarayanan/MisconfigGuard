"""
HybridRetriever — Weighted combination of semantic (FAISS) and keyword (BM25)
search over ingested code chunks.

Semantic search captures conceptual similarity; keyword search catches exact
security-critical terms ("0.0.0.0/0", "public-read", "AllowAll") that
embedding models under-weight.  Scores from both approaches are min-max
normalised then linearly combined:

    final_score = α · semantic_score + (1 − α) · keyword_score

By default α = 0.7 (semantic bias).

Architecture
------------
* ``VectorStoreManager``   — FAISS semantic search
* ``KeywordSearchEngine``  — BM25 / TF-IDF / built-in keyword search
* ``ThreadPoolExecutor``   — Parallel retrieval of semantic + keyword results

The keyword index is lazily synchronised from the vector store's SQLite
metadata store whenever the chunk count diverges, so no explicit sync call
is required after ingestion.

Usage
-----
    retriever = HybridRetriever(
        vector_store=vsm, embedder=emb,
        semantic_weight=0.7, keyword_weight=0.3,
    )
    results = retriever.retrieve("public S3 bucket", top_k=5)
    for r in results:
        print(r.final_score, r.chunk["chunk_id"])
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from keyword_search import KeywordSearchEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


@dataclass(order=True)
class RetrievalResult:
    """
    A single chunk returned by the hybrid retriever.

    Attributes
    ----------
    final_score :
        Weighted combination of semantic + keyword scores, in [0, 1].
        Used for sorting (highest first).
    chunk :
        Full chunk dict as stored in the vector/metadata store.
    semantic_score :
        Normalised similarity score from FAISS (0 if not found semantically).
    keyword_score :
        Normalised score from the keyword engine (0 if not found via keywords).
    rank :
        1-based position in the final ranked list.
    """

    final_score   : float
    chunk         : dict  = field(compare=False)
    semantic_score: float = field(compare=False, default=0.0)
    keyword_score : float = field(compare=False, default=0.0)
    rerank_score  : Optional[float] = field(compare=False, default=None)
    rank          : int   = field(compare=False, default=0)

    def to_dict(self) -> dict:
        result = {
            "chunk":          self.chunk,
            "final_score":    round(self.final_score,    4),
            "semantic_score": round(self.semantic_score, 4),
            "keyword_score":  round(self.keyword_score,  4),
            "rank":           self.rank,
        }
        if self.rerank_score is not None:
            result["rerank_score"] = round(self.rerank_score, 4)
        return result


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """
    Combine semantic (FAISS) and keyword (BM25) search with weighted fusion.

    Parameters
    ----------
    vector_store :
        A ``VectorStoreManager`` instance that already holds ingested chunks.
    embedder :
        An ``EmbeddingGenerator`` instance used to embed the text query.
    semantic_weight :
        Weight for the semantic similarity score (default 0.7).
    keyword_weight :
        Weight for the keyword relevance score (default 0.3).
        ``semantic_weight + keyword_weight`` should equal 1.0.
    max_workers :
        Number of parallel threads for retrieval (default 2).
    """

    def __init__(
        self,
        vector_store: Any,
        embedder:     Any,
        semantic_weight: float = 0.7,
        keyword_weight:  float = 0.3,
        rerank_embedder: Optional[Any] = None,
        rerank_top_k:    Optional[int] = None,
        max_workers:     int   = 2,
    ) -> None:
        if not (0.0 <= semantic_weight <= 1.0 and 0.0 <= keyword_weight <= 1.0):
            raise ValueError("Weights must be in [0, 1].")

        self.vector_store    = vector_store
        self.embedder        = embedder
        self.rerank_embedder = rerank_embedder
        self.rerank_top_k    = rerank_top_k
        self.semantic_weight = semantic_weight
        self.keyword_weight  = keyword_weight
        self.max_workers     = max_workers

        self._keyword_engine = KeywordSearchEngine()
        self._keyword_synced_count = -1  # tracks when keyword index was last built
        self._keyword_synced_next_id = -1

    # ------------------------------------------------------------------
    # Keyword index synchronisation
    # ------------------------------------------------------------------

    def _sync_keyword_index_if_needed(self) -> None:
        """
        Rebuild the keyword index from SQLite metadata if the vector store
        has grown (new chunks ingested) since the last build.
        """
        total = self.vector_store.total_vectors
        next_id = self._current_next_faiss_id()
        if (
            total == self._keyword_synced_count
            and next_id == self._keyword_synced_next_id
        ):
            return  # already up to date
        if total == 0:
            # Nothing to index; mark as synced so we don't retry every call.
            self._keyword_synced_count = 0
            self._keyword_synced_next_id = next_id
            return

        chunks = self._load_all_chunks()
        self._keyword_engine.index(chunks)
        self._keyword_synced_count = total
        self._keyword_synced_next_id = next_id
        logger.debug(
            "HybridRetriever: keyword index synced with %d chunks", total
        )

    def _current_next_faiss_id(self) -> int:
        """
        Return a monotonically increasing marker for metadata changes.

        For FAISS-backed stores, ``next_faiss_id`` changes whenever vectors are
        added or replaced (delete+add), even if total vector count is unchanged.
        """
        try:
            return int(self.vector_store.metadata_store.next_faiss_id())
        except Exception:
            # Non-FAISS or test doubles may not expose metadata_store.
            return self.vector_store.total_vectors

    def _load_all_chunks(self) -> List[dict]:
        """Fetch all chunk dicts from the vector store's metadata store."""
        try:
            # VectorStoreManager.metadata_store is a MetadataStore (SQLite)
            ms = self.vector_store.metadata_store
            rows = ms.query()              # returns all rows
            return [ms.to_chunk_dict(r) for r in rows]
        except Exception as exc:          # pragma: no cover
            logger.warning("Could not load chunks from metadata store: %s", exc)
            return []

    def force_sync(self) -> int:
        """
        Force rebuild of the keyword index from the current vector store.

        Returns the number of chunks indexed.
        """
        self._keyword_synced_count = -1
        self._keyword_synced_next_id = -1
        self._sync_keyword_index_if_needed()
        return self._keyword_engine.total_indexed

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query:           str,
        top_k:           int = 5,
        metadata_filter: Optional[Dict] = None,
        extra_k_factor:  int = 3,
    ) -> List[RetrievalResult]:
        """
        Retrieve the most relevant chunks for *query* using hybrid search.

        Parameters
        ----------
        query :
            Natural-language or keyword query string.
        top_k :
            Number of final results to return.
        metadata_filter :
            Optional dict for metadata pre-filtering in semantic search
            (e.g., ``{"cloud_provider": "aws"}``).
        extra_k_factor :
            Multiplier applied to *top_k* when fetching candidates from each
            backend before merging.  Higher values give better recall.

        Returns
        -------
        List[RetrievalResult]
            Ranked list (best first), capped at *top_k*.
        """
        self._sync_keyword_index_if_needed()

        fetch_k = max(top_k * extra_k_factor, 10)

        # Embed query once, reused for semantic search.
        query_embedding = self._embed_query(query)

        # Run semantic and keyword search in parallel.
        semantic_pairs: List[Tuple[dict, float]] = []
        keyword_pairs:  List[Tuple[dict, float]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._semantic_search, query_embedding, fetch_k, metadata_filter
                ): "semantic",
                executor.submit(
                    self._keyword_search, query, fetch_k
                ): "keyword",
            }
            for future in as_completed(futures):
                kind   = futures[future]
                result = future.result()
                if kind == "semantic":
                    semantic_pairs = result
                else:
                    keyword_pairs = result

        # Merge by chunk_id, keeping best scores from each backend.
        merged = self._merge(semantic_pairs, keyword_pairs)

        # Sort by final score descending, assign ranks, trim to top_k.
        ranked = sorted(merged.values(), key=lambda r: r.final_score, reverse=True)
        if self.rerank_embedder is not None and ranked:
            ranked = self._rerank(query, ranked, top_k=top_k, fetch_k=fetch_k)
        for idx, res in enumerate(ranked[:top_k], start=1):
            res.rank = idx

        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a text query and return a 1-D numpy array."""
        vecs = self.embedder.embed([query])
        return np.asarray(vecs[0], dtype=np.float32)

    def _rerank(
        self,
        query: str,
        ranked: List[RetrievalResult],
        top_k: int,
        fetch_k: int,
    ) -> List[RetrievalResult]:
        candidate_limit = min(
            len(ranked),
            max(top_k, self.rerank_top_k or fetch_k),
        )
        if candidate_limit <= 0:
            return ranked

        prefix = ranked[:candidate_limit]
        suffix = ranked[candidate_limit:]
        texts = [query] + [result.chunk.get("text", "") for result in prefix]
        embeddings = np.asarray(self.rerank_embedder.embed(texts), dtype=np.float32)
        query_vec = embeddings[0]
        doc_vecs = embeddings[1:]

        query_norm = float(np.linalg.norm(query_vec)) or 1e-10
        doc_norms = np.linalg.norm(doc_vecs, axis=1)
        doc_norms = np.where(doc_norms == 0, 1e-10, doc_norms)
        similarities = np.dot(doc_vecs, query_vec) / (doc_norms * query_norm)

        base_scores = {id(result): result.final_score for result in prefix}
        for result, similarity in zip(prefix, similarities):
            rerank_score = float((float(similarity) + 1.0) / 2.0)
            result.rerank_score = rerank_score
            result.final_score = rerank_score

        prefix = sorted(
            prefix,
            key=lambda result: (
                result.rerank_score if result.rerank_score is not None else -1.0,
                base_scores[id(result)],
            ),
            reverse=True,
        )
        logger.debug(
            "HybridRetriever reranked %d candidate(s) with %s",
            candidate_limit,
            getattr(self.rerank_embedder, "model_name", type(self.rerank_embedder).__name__),
        )
        return prefix + suffix

    def _semantic_search(
        self,
        query_embedding: np.ndarray,
        top_k:           int,
        metadata_filter: Optional[Dict],
    ) -> List[Tuple[dict, float]]:
        """Run FAISS similarity search; return (chunk, score) pairs."""
        try:
            results = self.vector_store.search(
                query_embedding, top_k=top_k, metadata_filter=metadata_filter
            )
            return [(r.chunk, r.score) for r in results]
        except Exception as exc:
            logger.warning("Semantic search failed: %s", exc)
            return []

    def _keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Tuple[dict, float]]:
        """Run keyword search; return (chunk, score) pairs."""
        try:
            return self._keyword_engine.search(query, top_k=top_k)
        except Exception as exc:
            logger.warning("Keyword search failed: %s", exc)
            return []

    def _chunk_key(self, chunk: dict) -> str:
        """Return a stable deduplication key for a chunk dict."""
        cid = chunk.get("chunk_id") or chunk.get("id")
        if cid:
            return str(cid)
        # Fallback: file_path + index
        return f"{chunk.get('file_path', '')}:{chunk.get('chunk_index', '')}"

    def _merge(
        self,
        semantic_pairs: List[Tuple[dict, float]],
        keyword_pairs:  List[Tuple[dict, float]],
    ) -> Dict[str, RetrievalResult]:
        """
        Union semantic and keyword candidates; compute weighted final score.

        Chunks that appear in only one backend receive a 0 for the missing score.
        """
        by_key: Dict[str, RetrievalResult] = {}

        for chunk, sem_score in semantic_pairs:
            key = self._chunk_key(chunk)
            if key not in by_key:
                by_key[key] = RetrievalResult(
                    final_score    = self.semantic_weight * sem_score,
                    chunk          = chunk,
                    semantic_score = sem_score,
                    keyword_score  = 0.0,
                )
            else:
                r = by_key[key]
                r.semantic_score = sem_score
                r.final_score    = (
                    self.semantic_weight * sem_score
                    + self.keyword_weight  * r.keyword_score
                )

        for chunk, kw_score in keyword_pairs:
            key = self._chunk_key(chunk)
            if key not in by_key:
                by_key[key] = RetrievalResult(
                    final_score    = self.keyword_weight * kw_score,
                    chunk          = chunk,
                    semantic_score = 0.0,
                    keyword_score  = kw_score,
                )
            else:
                r = by_key[key]
                r.keyword_score = kw_score
                r.final_score   = (
                    self.semantic_weight * r.semantic_score
                    + self.keyword_weight * kw_score
                )

        return by_key

    # ------------------------------------------------------------------
    # Index management helpers
    # ------------------------------------------------------------------

    def index_chunks(self, chunks: List[dict]) -> None:
        """
        Manually populate the keyword index with *chunks*.

        Useful when the vector store is not backed by a SQLite metadata
        store (e.g., in unit tests).
        """
        self._keyword_engine.index(chunks)
        self._keyword_synced_count = len(chunks)
        self._keyword_synced_next_id = -1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def keyword_engine(self) -> KeywordSearchEngine:
        """Direct access to the underlying keyword search engine."""
        return self._keyword_engine

    @property
    def total_indexed(self) -> int:
        """Number of chunks in the keyword index (may lag vector store)."""
        return self._keyword_engine.total_indexed
