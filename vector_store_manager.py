"""
VectorStoreManager — Enhanced vector storage with SQLite metadata, CRUD
operations, cloud-provider detection, and rich retrieval for MisconfigGuard.

Backends
--------
faiss  (default)
    In-memory FAISS IndexIDMap(IndexFlatIP) — inner product = cosine after
    L2-norm.  Supports explicit integer IDs so individual chunks can be
    updated or deleted without rebuilding the whole index.

    Persisted to:
        <index_path>.faiss   — FAISS binary index
        <index_path>.db      — SQLite metadata store

    Backward-compat: if an old <index_path>.chunks.pkl is found on load,
    it is automatically migrated to SQLite and the .pkl file is removed.

chroma
    ChromaDB persistent collection.  Auto-persists on every write.
    Metadata is stored inside the Chroma document store.
    Requires:  pip install chromadb

Architecture
------------
    VectorStoreManager
    ├── MetadataStore (SQLite)  — chunk metadata, content hash, cloud provider
    └── FAISS IndexIDMap        — vector similarity; IDs are int64 foreign keys
                                  into the SQLite `faiss_id` column

Key new APIs
------------
    create_index(dim)                          — explicit index initialisation
    add_embeddings(embeddings, chunks)         — hash-based deduplication
    update_embeddings(chunk_ids, embs, chunks) — atomic delete + re-add
    delete_embeddings(chunk_ids)               — remove vectors + metadata
    similarity_search(query_emb, k, filter)    — metadata-filtered search
    save_index(path) / load_index(path)        — persistence with optional path

Backward-compatible aliases
---------------------------
    add(embeddings, chunks)          → add_embeddings(embeddings, chunks)
    search(query_embedding, top_k)   → similarity_search(query_embedding, top_k)
    save() / load()                  — unchanged signatures
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cloud-provider detection
# ---------------------------------------------------------------------------

# Maps resource-type prefix (first component before "_") to provider name.
_PROVIDER_PREFIX_MAP: Dict[str, str] = {
    "aws":        "aws",
    "azurerm":    "azure",
    "azuread":    "azure",
    "azurestack": "azure",
    "google":     "gcp",
    "kubernetes": "k8s",
    "helm":       "k8s",
}

# Keyword patterns used when the prefix heuristic fails.
_PROVIDER_KEYWORDS: List[tuple] = [
    ("aws",       ["amazonaws", '"aws_', "aws-", "us-east-", "us-west-", "eu-west-"]),
    ("azure",     ["azure", "microsoft", "azurewebsites", "azureedge"]),
    ("gcp",       ["googleapis", '"google_', "gcp", "gcloud", "cloudfunctions"]),
    ("k8s",       ["kubernetes", "k8s", "kubectl", "apiVersion", "kind:"]),
]


def detect_cloud_provider(
    resource_type: str = "",
    content: str = "",
    file_path: str = "",
) -> str:
    """
    Infer the cloud provider from a resource type prefix, file content,
    or file path.

    Returns one of: "aws" | "azure" | "gcp" | "k8s" | "unknown"
    """
    if resource_type:
        prefix = resource_type.split("_")[0].lower()
        if prefix in _PROVIDER_PREFIX_MAP:
            return _PROVIDER_PREFIX_MAP[prefix]

    probe = (content + " " + file_path).lower()
    for provider, keywords in _PROVIDER_KEYWORDS:
        if any(kw.lower() in probe for kw in keywords):
            return provider

    return "unknown"


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single retrieval result ranked by similarity score."""

    chunk: dict    # Enriched chunk dict (includes metadata, cloud_provider, …)
    score: float   # Cosine similarity ∈ [0, 1] (higher = more relevant)
    rank: int      # 1-based rank in the result list

    def __repr__(self) -> str:
        return (
            f"SearchResult(rank={self.rank}, score={self.score:.4f}, "
            f"file={self.chunk.get('file_path', '')})"
        )


# ---------------------------------------------------------------------------
# MetadataStore — SQLite-backed chunk metadata and FAISS-ID mapping
# ---------------------------------------------------------------------------


class MetadataStore:
    """
    SQLite metadata store that runs alongside the FAISS vector index.

    Each row represents one chunk vector.  The ``faiss_id`` column is the
    explicit int64 ID used in FAISS ``IndexIDMap.add_with_ids()``, enabling
    O(1) chunk lookup after a vector search.

    Schema
    ------
    chunks:
        chunk_id       TEXT PRIMARY KEY   — deterministic semantic ID
        faiss_id       INTEGER UNIQUE     — int64 key in FAISS IndexIDMap
        file_path      TEXT
        file_type      TEXT
        chunk_index    INTEGER
        resource_type  TEXT
        cloud_provider TEXT
        content        TEXT               — chunk text
        content_hash   TEXT               — SHA-256 for deduplication
        tokens         INTEGER
        dependencies   TEXT               — JSON-serialised list
        repo           TEXT
        timestamp      TEXT               — ISO-8601 UTC
        metadata_json  TEXT               — full original metadata dict
    """

    _CREATE_SQL = """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id       TEXT    PRIMARY KEY,
            faiss_id       INTEGER UNIQUE,
            file_path      TEXT    NOT NULL DEFAULT '',
            file_type      TEXT    NOT NULL DEFAULT '',
            chunk_index    INTEGER NOT NULL DEFAULT 0,
            resource_type  TEXT    NOT NULL DEFAULT '',
            cloud_provider TEXT    NOT NULL DEFAULT 'unknown',
            content        TEXT    NOT NULL DEFAULT '',
            content_hash   TEXT    NOT NULL DEFAULT '',
            tokens         INTEGER NOT NULL DEFAULT 0,
            dependencies   TEXT    NOT NULL DEFAULT '[]',
            repo           TEXT    NOT NULL DEFAULT '',
            timestamp      TEXT    NOT NULL DEFAULT '',
            metadata_json  TEXT    NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_file_path    ON chunks(file_path);
        CREATE INDEX IF NOT EXISTS idx_content_hash ON chunks(content_hash);
        CREATE INDEX IF NOT EXISTS idx_cloud        ON chunks(cloud_provider);
        CREATE INDEX IF NOT EXISTS idx_resource     ON chunks(resource_type);
        CREATE INDEX IF NOT EXISTS idx_faiss_id     ON chunks(faiss_id);
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(self._CREATE_SQL)

    @contextmanager
    def _connect(self) -> Generator:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads
        conn.execute("PRAGMA synchronous=NORMAL")  # balance safety/speed
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- Sequencing ---------------------------------------------------------

    def next_faiss_id(self) -> int:
        """Return the next available FAISS int64 ID (monotonically increasing)."""
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(faiss_id) FROM chunks").fetchone()
            current = row[0]
        return 0 if current is None else int(current) + 1

    # ---- CRUD ---------------------------------------------------------------

    def insert(self, chunk_id: str, faiss_id: int, chunk: dict) -> None:
        """Insert or replace a chunk row."""
        content: str = chunk.get("text", "")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        meta: dict = chunk.get("metadata", {})

        # Extract resource_type from multiple possible locations.
        resource_type: str = (
            chunk.get("resource_type")
            or meta.get("resource_type", "")
            or meta.get("block_type", "")
            or ""
        )
        cloud_provider = (
            chunk.get("cloud_provider")
            or meta.get("cloud_provider")
            or detect_cloud_provider(
                resource_type=resource_type,
                content=content,
                file_path=chunk.get("file_path", ""),
            )
        )
        timestamp: str = chunk.get("timestamp") or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id, faiss_id, file_path, file_type, chunk_index,
                    resource_type, cloud_provider, content, content_hash,
                    tokens, dependencies, repo, timestamp, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    faiss_id,
                    chunk.get("file_path", ""),
                    chunk.get("file_type", ""),
                    chunk.get("chunk_index", 0),
                    resource_type,
                    cloud_provider,
                    content,
                    content_hash,
                    chunk.get("tokens", 0),
                    json.dumps(chunk.get("dependencies", [])),
                    meta.get("repo", ""),
                    timestamp,
                    json.dumps(meta),
                ),
            )

    def delete(self, chunk_ids: List[str]) -> List[int]:
        """
        Delete rows by chunk_id and return the freed FAISS IDs.

        The caller must remove the corresponding vectors from FAISS.
        """
        if not chunk_ids:
            return []
        ph = ",".join("?" * len(chunk_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT faiss_id FROM chunks WHERE chunk_id IN ({ph})",
                chunk_ids,
            ).fetchall()
            freed = [int(r["faiss_id"]) for r in rows if r["faiss_id"] is not None]
            conn.execute(
                f"DELETE FROM chunks WHERE chunk_id IN ({ph})",
                chunk_ids,
            )
        return freed

    def delete_by_file(self, file_path: str) -> List[int]:
        """Delete all chunks belonging to *file_path*; return freed FAISS IDs."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT faiss_id FROM chunks WHERE file_path = ?",
                (file_path,),
            ).fetchall()
            freed = [int(r["faiss_id"]) for r in rows if r["faiss_id"] is not None]
            conn.execute(
                "DELETE FROM chunks WHERE file_path = ?", (file_path,)
            )
        return freed

    # ---- Lookups ------------------------------------------------------------

    def get(self, chunk_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_by_faiss_id(self, faiss_id: int) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE faiss_id = ?", (faiss_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_chunk_ids_for_file(self, file_path: str) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE file_path = ?", (file_path,)
            ).fetchall()
        return [r["chunk_id"] for r in rows]

    def content_hash_exists(self, content_hash: str) -> bool:
        """Return True if an identical chunk (same content hash) is stored."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        return row is not None

    def chunk_id_for_hash(self, content_hash: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT chunk_id FROM chunks WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        return row["chunk_id"] if row else None

    # ---- Bulk queries -------------------------------------------------------

    def query(
        self,
        faiss_ids: Optional[List[int]] = None,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """
        Return rows matching *faiss_ids* and/or *metadata_filter*.

        *metadata_filter* supports equality checks on:
            file_path, file_type, resource_type, cloud_provider, repo
        """
        _filterable = {
            "file_path", "file_type", "resource_type", "cloud_provider", "repo"
        }
        conditions: List[str] = []
        params: List[Any] = []

        if faiss_ids is not None:
            ph = ",".join("?" * len(faiss_ids))
            conditions.append(f"faiss_id IN ({ph})")
            params.extend(faiss_ids)

        if metadata_filter:
            for key, val in metadata_filter.items():
                if key in _filterable:
                    conditions.append(f"{key} = ?")
                    params.append(val)

        sql = "SELECT * FROM chunks"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ---- Serialisation helpers ----------------------------------------------

    def to_chunk_dict(self, row: Dict) -> Dict:
        """Convert a SQLite row into the enriched chunk dict format."""
        # Deserialise JSON fields safely.
        try:
            meta = json.loads(row.get("metadata_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            meta = {}
        try:
            deps = json.loads(row.get("dependencies") or "[]")
        except (json.JSONDecodeError, TypeError):
            deps = []

        enriched_meta = {
            **meta,
            "resource_type":  row.get("resource_type", ""),
            "cloud_provider": row.get("cloud_provider", "unknown"),
            "repo":           row.get("repo", ""),
            "timestamp":      row.get("timestamp", ""),
        }
        return {
            "chunk_id":      row.get("chunk_id", ""),
            "text":          row.get("content", ""),
            "file_path":     row.get("file_path", ""),
            "file_type":     row.get("file_type", ""),
            "chunk_index":   row.get("chunk_index", 0),
            "tokens":        row.get("tokens", 0),
            "resource_type": row.get("resource_type", ""),
            "cloud_provider": row.get("cloud_provider", "unknown"),
            "dependencies":  deps,
            "timestamp":     row.get("timestamp", ""),
            "metadata":      enriched_meta,
        }

    # ---- Properties ---------------------------------------------------------

    @property
    def total(self) -> int:
        with self._connect() as conn:
            return int(
                conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _auto_chunk_id(chunk: dict, index: int = 0) -> str:
    """Generate a deterministic chunk_id for legacy chunks that lack one."""
    stem = Path(chunk.get("file_path", "chunk")).stem
    idx  = chunk.get("chunk_index", index)
    return f"{stem}_{idx}"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return vecs / norms


# ---------------------------------------------------------------------------
# VectorStoreManager
# ---------------------------------------------------------------------------


class VectorStoreManager:
    """
    Unified interface over FAISS and Chroma vector backends with a
    SQLite-backed metadata store for rich filtering and deduplication.
    """

    def __init__(
        self,
        backend: str = "faiss",
        index_path: str = "./cache/faiss_index",
        chroma_persist_dir: str = "./cache/chroma",
        embedding_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            backend:             "faiss" or "chroma".
            index_path:          Base path for FAISS files (no extension).
                                 SQLite lives at <index_path>.db.
            chroma_persist_dir:  Directory for ChromaDB persistence.
            embedding_dim:       Pre-declare embedding dimension (optional;
                                 inferred from the first add() call).
        """
        if backend not in ("faiss", "chroma"):
            raise ValueError(
                f"Unknown backend '{backend}'. Choose 'faiss' or 'chroma'."
            )
        self.backend    = backend
        self.index_path = Path(index_path)
        self.chroma_persist_dir = chroma_persist_dir
        self.embedding_dim = embedding_dim

        # FAISS state
        self._faiss_index = None
        self._meta: MetadataStore = MetadataStore(str(self.index_path) + ".db")

        # Chroma state
        self._chroma_collection = None

    # ==================================================================
    # FAISS backend — private
    # ==================================================================

    def _faiss_import(self):
        try:
            import faiss
            return faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is not installed.  Run:  pip install faiss-cpu"
            )

    def _faiss_ensure(self, dim: int) -> None:
        """Initialise IndexIDMap(IndexFlatIP) if not already created."""
        if self._faiss_index is not None:
            return
        faiss = self._faiss_import()
        flat = faiss.IndexFlatIP(dim)
        self._faiss_index = faiss.IndexIDMap(flat)
        self.embedding_dim = dim
        logger.info("FAISS IndexIDMap(IndexFlatIP) initialised (dim=%d)", dim)

    def _faiss_add_with_ids(
        self, embeddings: np.ndarray, faiss_ids: List[int]
    ) -> None:
        """Add L2-normalised *embeddings* with explicit *faiss_ids*."""
        self._faiss_ensure(embeddings.shape[1])
        normed = _l2_normalize(embeddings.astype(np.float32))
        ids_arr = np.array(faiss_ids, dtype=np.int64)
        self._faiss_index.add_with_ids(normed, ids_arr)

    def _faiss_remove_ids(self, faiss_ids: List[int]) -> int:
        """
        Remove vectors by FAISS ID.

        Returns the number of vectors actually removed.
        Handles older and newer faiss-cpu swig API variants.
        """
        if not faiss_ids or self._faiss_index is None:
            return 0
        faiss = self._faiss_import()
        ids_arr = np.array(faiss_ids, dtype=np.int64)
        try:
            # Newer faiss Python bindings accept an ndarray directly.
            sel = faiss.IDSelectorArray(ids_arr)
        except TypeError:
            # Older bindings require (count, pointer).
            sel = faiss.IDSelectorArray(
                len(ids_arr), faiss.swig_ptr(ids_arr)
            )
        return self._faiss_index.remove_ids(sel)

    def _faiss_search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[SearchResult]:
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []

        # If a metadata filter is requested, pre-fetch the matching faiss_ids
        # from SQLite and build an ID-filtered result set.
        if metadata_filter:
            return self._faiss_filtered_search(query_vec, top_k, metadata_filter)

        normed = _l2_normalize(
            query_vec.reshape(1, -1).astype(np.float32)
        )
        k = min(top_k, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(normed, k)

        results: List[SearchResult] = []
        for rank, (faiss_id, score) in enumerate(
            zip(indices[0], scores[0])
        ):
            if faiss_id < 0:
                continue
            row = self._meta.get_by_faiss_id(int(faiss_id))
            if row is None:
                continue
            results.append(
                SearchResult(
                    chunk=self._meta.to_chunk_dict(row),
                    score=float(score),
                    rank=rank + 1,
                )
            )
        return results

    def _faiss_filtered_search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        metadata_filter: Dict[str, str],
    ) -> List[SearchResult]:
        """
        Metadata-filtered search: retrieve candidates from SQLite then
        score with FAISS by reconstructing their vectors.

        Works well for small-to-medium filtered sets (<50 k vectors).
        For very large filtered sets, a pre-filtering approach would be
        used instead; that is a recommended future optimisation.
        """
        faiss = self._faiss_import()
        candidate_rows = self._meta.query(metadata_filter=metadata_filter)
        if not candidate_rows:
            return []

        # Reconstruct vectors for matching FAISS IDs.
        candidate_faiss_ids = [
            r["faiss_id"] for r in candidate_rows
            if r["faiss_id"] is not None
        ]
        if not candidate_faiss_ids:
            return []

        # Reconstruct a temporary flat index for candidate-only search.
        dim = self.embedding_dim or self._faiss_index.d
        tmp_flat = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

        id_arr = np.array(candidate_faiss_ids, dtype=np.int64)
        vecs   = np.zeros((len(candidate_faiss_ids), dim), dtype=np.float32)
        for i, fid in enumerate(candidate_faiss_ids):
            try:
                self._faiss_index.reconstruct(int(fid), vecs[i])
            except Exception:
                pass  # vector may have been deleted; skip silently
        tmp_flat.add_with_ids(vecs, id_arr)

        normed = _l2_normalize(query_vec.reshape(1, -1).astype(np.float32))
        k = min(top_k, tmp_flat.ntotal)
        scores, indices = tmp_flat.search(normed, k)

        results: List[SearchResult] = []
        for rank, (faiss_id, score) in enumerate(
            zip(indices[0], scores[0])
        ):
            if faiss_id < 0:
                continue
            row = self._meta.get_by_faiss_id(int(faiss_id))
            if row is None:
                continue
            results.append(
                SearchResult(
                    chunk=self._meta.to_chunk_dict(row),
                    score=float(score),
                    rank=rank + 1,
                )
            )
        return results

    def _faiss_save(self) -> None:
        if self._faiss_index is None:
            return
        faiss = self._faiss_import()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(
            self._faiss_index, str(self.index_path) + ".faiss"
        )
        logger.info(
            "FAISS index saved (%d vectors) -> %s.faiss",
            self._faiss_index.ntotal,
            self.index_path,
        )

    def _faiss_load(self) -> bool:
        faiss = self._faiss_import()
        idx_file    = str(self.index_path) + ".faiss"
        pkl_file    = str(self.index_path) + ".chunks.pkl"

        if not Path(idx_file).exists():
            return False

        self._faiss_index = faiss.read_index(idx_file)
        logger.info(
            "FAISS index loaded (%d vectors) <- %s.faiss",
            self._faiss_index.ntotal,
            self.index_path,
        )

        # Migrate legacy .chunks.pkl → SQLite if SQLite is empty.
        if Path(pkl_file).exists() and self._meta.total == 0:
            self._migrate_pkl_to_sqlite(pkl_file)

        return True

    def _migrate_pkl_to_sqlite(self, pkl_file: str) -> None:
        """
        One-time migration: read old .chunks.pkl and populate SQLite.

        The old index used positional IDs (0, 1, …, n-1).  We rebuild it
        as an IndexIDMap with the same IDs and migrate metadata to SQLite.
        """
        try:
            with open(pkl_file, "rb") as fh:
                old_chunks: List[dict] = pickle.load(fh)
        except Exception as exc:
            logger.warning("Migration failed to read %s: %s", pkl_file, exc)
            return

        faiss = self._faiss_import()
        n   = len(old_chunks)
        dim = self._faiss_index.d

        # Reconstruct vectors from the old IndexFlatIP.
        vecs = np.zeros((n, dim), dtype=np.float32)
        try:
            self._faiss_index.reconstruct_n(0, n, vecs)
        except Exception as exc:
            logger.warning("Cannot reconstruct old vectors: %s", exc)
            return

        # Build a new IndexIDMap with explicit IDs.
        flat  = faiss.IndexFlatIP(dim)
        new_idx = faiss.IndexIDMap(flat)
        ids   = np.arange(n, dtype=np.int64)
        new_idx.add_with_ids(_l2_normalize(vecs), ids)
        self._faiss_index = new_idx
        self.embedding_dim = dim

        # Populate SQLite.
        for i, chunk in enumerate(old_chunks):
            chunk_id = (
                chunk.get("chunk_id") or _auto_chunk_id(chunk, i)
            )
            self._meta.insert(chunk_id, i, chunk)

        # Remove the old pickle file.
        try:
            Path(pkl_file).unlink()
        except OSError:
            pass

        logger.info(
            "Migrated %d chunks from legacy .chunks.pkl → SQLite", n
        )

    # ==================================================================
    # Chroma backend — private
    # ==================================================================

    def _chroma_ensure(self) -> None:
        if self._chroma_collection is not None:
            return
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is not installed.  Run:  pip install chromadb"
            )
        client = chromadb.PersistentClient(path=self.chroma_persist_dir)
        self._chroma_collection = client.get_or_create_collection(
            name="misconfigguard",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Chroma collection ready (%d existing vectors)",
            self._chroma_collection.count(),
        )

    def _chroma_add(self, embeddings: np.ndarray, chunks: List[dict]) -> None:
        self._chroma_ensure()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        start = self._chroma_collection.count()
        ids      = [str(start + i) for i in range(len(chunks))]
        docs     = [c.get("text", "") for c in chunks]
        metadatas = []
        for c in chunks:
            meta = c.get("metadata", {})
            resource_type = (
                c.get("resource_type")
                or meta.get("resource_type", "")
                or ""
            )
            cloud_provider = detect_cloud_provider(
                resource_type=resource_type,
                content=c.get("text", ""),
                file_path=c.get("file_path", ""),
            )
            metadatas.append({
                "chunk_id":      c.get("chunk_id", str(start + len(metadatas))),
                "file_path":     c.get("file_path", ""),
                "file_type":     c.get("file_type", ""),
                "chunk_index":   str(c.get("chunk_index", 0)),
                "resource_type": resource_type,
                "cloud_provider": cloud_provider,
                "timestamp":     timestamp,
                **{k: str(v) for k, v in meta.items() if isinstance(v, (str, int, float, bool))},
            })
        self._chroma_collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=docs,
            metadatas=metadatas,
        )

    def _chroma_search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[SearchResult]:
        self._chroma_ensure()
        n = min(top_k, self._chroma_collection.count())
        if n == 0:
            return []
        where = (
            {k: v for k, v in metadata_filter.items()}
            if metadata_filter
            else None
        )
        kwargs: dict = dict(
            query_embeddings=[query_vec.tolist()],
            n_results=n,
        )
        if where:
            kwargs["where"] = where

        results = self._chroma_collection.query(**kwargs)
        out: List[SearchResult] = []
        for rank, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            enriched_meta = {
                k: v for k, v in meta.items()
                if k not in ("file_path", "file_type", "chunk_index",
                             "chunk_id", "cloud_provider", "resource_type",
                             "timestamp")
            }
            chunk = {
                "chunk_id":      meta.get("chunk_id", ""),
                "text":          doc,
                "file_path":     meta.get("file_path", ""),
                "file_type":     meta.get("file_type", ""),
                "chunk_index":   int(meta.get("chunk_index", 0)),
                "tokens":        int(meta.get("tokens", 0)),
                "resource_type": meta.get("resource_type", ""),
                "cloud_provider": meta.get("cloud_provider", "unknown"),
                "dependencies":  [],
                "timestamp":     meta.get("timestamp", ""),
                "metadata": {
                    **enriched_meta,
                    "resource_type":  meta.get("resource_type", ""),
                    "cloud_provider": meta.get("cloud_provider", "unknown"),
                    "timestamp":      meta.get("timestamp", ""),
                },
            }
            # Chroma cosine distance [0, 2] → similarity [−1, 1].
            out.append(
                SearchResult(chunk, max(0.0, 1.0 - float(dist)), rank + 1)
            )
        return out

    # ==================================================================
    # New public API — CRUD + metadata-filtered search
    # ==================================================================

    def create_index(self, dim: int) -> None:
        """
        Explicitly create the FAISS index for *dim*-dimensional embeddings.

        Called automatically by ``add_embeddings()`` if not already created.
        Useful when you want to pre-allocate the index before ingestion.
        """
        if self.backend == "faiss":
            self._faiss_ensure(dim)

    def add_embeddings(
        self, embeddings: np.ndarray, chunks: List[dict]
    ) -> int:
        """
        Add *embeddings* to the store, skipping chunks whose content has
        already been indexed (SHA-256 hash deduplication).

        Args:
            embeddings: Float32 array of shape (N, D).
            chunks:     List of N chunk dicts (IntelligentChunk.to_dict() or
                        legacy Chunk.to_dict() format).

        Returns:
            Number of NEW chunks actually added (duplicates excluded).
        """
        if len(embeddings) == 0:
            return 0

        new_embs:    List[np.ndarray] = []
        new_faiss_ids: List[int]      = []
        new_chunks:  List[dict]       = []

        if self.backend == "faiss":
            next_id = self._meta.next_faiss_id()
            for i, (emb, chunk) in enumerate(zip(embeddings, chunks)):
                content = chunk.get("text", "")
                chash   = _content_hash(content)
                if self._meta.content_hash_exists(chash):
                    continue  # Skip exact duplicate
                chunk_id = (
                    chunk.get("chunk_id") or _auto_chunk_id(chunk, i)
                )
                faiss_id = next_id + len(new_embs)
                self._meta.insert(chunk_id, faiss_id, chunk)
                new_embs.append(emb)
                new_faiss_ids.append(faiss_id)
                new_chunks.append(chunk)

            if new_embs:
                self._faiss_add_with_ids(
                    np.vstack(new_embs), new_faiss_ids
                )
        else:
            # Chroma: pass through (Chroma handles its own dedup via IDs).
            self._chroma_add(embeddings, chunks)
            return len(chunks)

        logger.debug("add_embeddings: %d new, %d skipped (duplicate)",
                     len(new_embs), len(chunks) - len(new_embs))
        return len(new_embs)

    def update_embeddings(
        self,
        chunk_ids: List[str],
        embeddings: np.ndarray,
        new_chunks: List[dict],
    ) -> int:
        """
        Atomically replace the embeddings and metadata for *chunk_ids*.

        Implementation: delete old vectors → add new ones.

        Args:
            chunk_ids:   Existing chunk IDs to replace.
            embeddings:  New embeddings, shape (len(chunk_ids), D).
            new_chunks:  New chunk dicts matching *chunk_ids* positionally.

        Returns:
            Number of vectors added (≤ len(chunk_ids)).
        """
        self.delete_embeddings(chunk_ids)
        return self.add_embeddings(embeddings, new_chunks)

    def delete_embeddings(self, chunk_ids: List[str]) -> int:
        """
        Remove *chunk_ids* from the vector store and metadata store.

        Args:
            chunk_ids: List of chunk_id strings to remove.

        Returns:
            Number of vectors actually removed from FAISS.
        """
        if not chunk_ids:
            return 0

        if self.backend == "faiss":
            freed_faiss_ids = self._meta.delete(chunk_ids)
            removed = self._faiss_remove_ids(freed_faiss_ids)
            logger.debug(
                "delete_embeddings: removed %d vectors (requested %d chunk_ids)",
                removed, len(chunk_ids),
            )
            return removed
        else:
            logger.warning(
                "delete_embeddings is not fully supported for Chroma backend."
            )
            return 0

    def delete_file(self, file_path: str) -> int:
        """
        Remove all chunks belonging to *file_path*.

        Convenience wrapper used by the incremental indexer when a file
        has been modified or deleted from the repository.

        Returns:
            Number of vectors removed.
        """
        if self.backend == "faiss":
            freed = self._meta.delete_by_file(file_path)
            return self._faiss_remove_ids(freed)
        return 0

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[SearchResult]:
        """
        Return the *k* most similar chunks, optionally filtered by metadata.

        Args:
            query_embedding: 1-D float32 query vector.
            k:               Maximum number of results to return.
            metadata_filter: Optional dict restricting results to chunks
                             whose metadata fields match exactly.  Supported
                             keys: file_path, file_type, resource_type,
                             cloud_provider, repo.

        Returns:
            List of SearchResult objects, ranked by descending similarity.

        Example:
            # Only return AWS resources
            results = store.similarity_search(
                query_vec, k=5,
                metadata_filter={"cloud_provider": "aws"}
            )
        """
        if self.backend == "faiss":
            return self._faiss_search(query_embedding, k, metadata_filter)
        return self._chroma_search(query_embedding, k, metadata_filter)

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """
        Retrieve a chunk dict by its chunk_id (no embedding comparison).

        Returns None if the chunk is not found.
        """
        if self.backend != "faiss":
            return None
        row = self._meta.get(chunk_id)
        return self._meta.to_chunk_dict(row) if row else None

    def get_chunks_for_file(self, file_path: str) -> List[dict]:
        """Return all chunks belonging to *file_path*."""
        if self.backend != "faiss":
            return []
        rows = self._meta.query(metadata_filter={"file_path": file_path})
        return [self._meta.to_chunk_dict(r) for r in rows]

    # ==================================================================
    # Backward-compatible aliases
    # ==================================================================

    def add(self, embeddings: np.ndarray, chunks: List[dict]) -> None:
        """Backward-compatible alias for ``add_embeddings()``."""
        self.add_embeddings(embeddings, chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[SearchResult]:
        """Backward-compatible alias for ``similarity_search()``."""
        return self.similarity_search(query_embedding, k=top_k, metadata_filter=metadata_filter)

    # ==================================================================
    # Persistence
    # ==================================================================

    def save_index(self, path: Optional[str] = None) -> None:
        """
        Persist the index to disk.

        Args:
            path: Optional override for the base path (no extension).
                  Defaults to the ``index_path`` set at construction time.
        """
        if path is not None:
            # Temporarily switch the path for this save.
            original = self.index_path
            self.index_path = Path(path)
            self._faiss_save()
            self.index_path = original
        else:
            if self.backend == "faiss" and self._faiss_index is not None:
                self._faiss_save()
        # SQLite is updated on every write; no explicit flush needed.

    def load_index(self, path: Optional[str] = None) -> bool:
        """
        Load the index from disk.

        Args:
            path: Optional override for the base path.

        Returns:
            True if the index was successfully loaded.
        """
        if path is not None:
            original = self.index_path
            self.index_path = Path(path)
            self._meta = MetadataStore(str(self.index_path) + ".db")
            result = self._faiss_load()
            self.index_path = original
            return result
        return self._faiss_load()

    def save(self) -> None:
        """Backward-compatible alias for ``save_index()``."""
        self.save_index()

    def has_persisted_state(self) -> bool:
        """Return True if on-disk vector-store artifacts exist."""
        if self.backend == "faiss":
            return any(
                Path(str(self.index_path) + suffix).exists()
                for suffix in (".faiss", ".db", ".chunks.pkl")
            )
        return Path(self.chroma_persist_dir).exists()

    def reset_persistence(self) -> None:
        """Remove persisted artifacts and reset in-memory state."""
        if self.backend == "faiss":
            for suffix in (".faiss", ".db", ".chunks.pkl"):
                path = Path(str(self.index_path) + suffix)
                try:
                    if path.exists():
                        path.unlink()
                except OSError:
                    logger.warning("Failed to remove stale vector-store artifact: %s", path)
            self._faiss_index = None
            self._meta = MetadataStore(str(self.index_path) + ".db")
            self.embedding_dim = None
            return

        self._chroma_collection = None

    def load(self) -> bool:
        """
        Backward-compatible alias for ``load_index()``.

        For Chroma, auto-loads when the collection is first opened.
        """
        if self.backend == "faiss":
            return self.load_index()
        # Chroma auto-loads on open.
        self._chroma_ensure()
        return self._chroma_collection.count() > 0

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def total_vectors(self) -> int:
        """Total number of vectors currently in the store."""
        if self.backend == "faiss":
            return self._faiss_index.ntotal if self._faiss_index else 0
        if self._chroma_collection:
            return self._chroma_collection.count()
        return 0

    @property
    def metadata_store(self) -> Optional[MetadataStore]:
        """Direct access to the SQLite metadata store (FAISS backend only)."""
        return self._meta if self.backend == "faiss" else None

