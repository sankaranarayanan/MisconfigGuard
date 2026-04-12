"""
EmbeddingGenerator — Produces dense vector embeddings for text chunks
using a locally-running sentence-transformers model.

Embeddings are cached to disk (per-model, per-text SHA-256 key) so that
re-indexing an unchanged file does not re-run the neural model while still
avoiding stale cache collisions after model changes.

Recommended models:
    sentence-transformers/all-MiniLM-L6-v2  (fast, 384-dim)
    BAAI/bge-large-en                        (higher accuracy, 1024-dim)
"""

import hashlib
from contextlib import contextmanager
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates and caches text embeddings using a local sentence-transformers
    model.  The model is loaded lazily on the first ``embed()`` call.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str = "./cache/embeddings",
        batch_size: int = 32,
    ):
        """
        Args:
            model_name: HuggingFace model identifier or local path.
            cache_dir:  Directory for per-text embedding cache files.
            batch_size: Number of texts sent to the model in one forward pass.
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self._model = None  # lazy-loaded

    @contextmanager
    def _quiet_model_load(self):
        """Temporarily suppress noisy third-party model-load logs and progress bars."""
        logger_names = (
            "httpx",
            "httpcore",
            "huggingface_hub",
            "transformers",
            "sentence_transformers",
        )
        original_logger_levels = {
            name: logging.getLogger(name).level for name in logger_names
        }

        transformers_logging = None
        transformers_verbosity = None
        transformers_progress_enabled = None
        hf_logging = None
        hf_verbosity = None
        hf_progress_disabled = None

        try:
            for name in logger_names:
                logging.getLogger(name).setLevel(logging.ERROR)

            try:
                from transformers.utils import logging as transformers_logging

                transformers_verbosity = transformers_logging.get_verbosity()
                transformers_progress_enabled = transformers_logging.is_progress_bar_enabled()
                transformers_logging.set_verbosity_error()
                transformers_logging.disable_progress_bar()
            except Exception:
                transformers_logging = None

            try:
                from huggingface_hub.utils import (
                    are_progress_bars_disabled,
                    disable_progress_bars,
                    enable_progress_bars,
                )
                from huggingface_hub.utils import logging as hf_logging

                hf_verbosity = hf_logging.get_verbosity()
                hf_progress_disabled = are_progress_bars_disabled()
                hf_logging.set_verbosity_error()
                if not hf_progress_disabled:
                    disable_progress_bars()
            except Exception:
                hf_logging = None
                disable_progress_bars = None
                enable_progress_bars = None

            yield
        finally:
            for name, level in original_logger_levels.items():
                logging.getLogger(name).setLevel(level)

            if transformers_logging is not None and transformers_verbosity is not None:
                transformers_logging.set_verbosity(transformers_verbosity)
                if transformers_progress_enabled:
                    transformers_logging.enable_progress_bar()
                else:
                    transformers_logging.disable_progress_bar()

            if hf_logging is not None and hf_verbosity is not None:
                hf_logging.set_verbosity(hf_verbosity)
                if hf_progress_disabled:
                    disable_progress_bars()
                else:
                    enable_progress_bars()

    # ------------------------------------------------------------------
    # Lazy model access
    # ------------------------------------------------------------------

    @property
    def model(self):
        """Load the SentenceTransformer model on first use."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            try:
                with self._quiet_model_load():
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed.\n"
                    "Run:  pip install sentence-transformers"
                )
            logger.info(
                "Model loaded (embedding dim=%d)", self.embedding_dim
            )
        return self._model

    # ------------------------------------------------------------------
    # Disk cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        payload = f"{self.model_name}\0{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def _load_cached(self, key: str) -> Optional[np.ndarray]:
        path = self._cache_path(key)
        if path.exists():
            try:
                with open(path, "rb") as fh:
                    return pickle.load(fh)
            except Exception as exc:
                logger.warning("Cache read error (%s): %s", key[:8], exc)
        return None

    def _save_cached(self, key: str, embedding: np.ndarray) -> None:
        try:
            with open(self._cache_path(key), "wb") as fh:
                pickle.dump(embedding, fh)
        except Exception as exc:
            logger.warning("Cache write error (%s): %s", key[:8], exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Return an (N, D) embedding matrix for *texts*.

        Cache-hits are served from disk; only uncached texts are sent
        to the model.  Results are written back to the cache.
        """
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Separate cached from uncached
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            cached = self._load_cached(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        logger.info(
            "Embedding batch: total_texts=%d cached=%d new=%d model=%s",
            len(texts),
            len(texts) - len(uncached_texts),
            len(uncached_texts),
            self.model_name,
        )

        # Batch-encode uncached texts
        if uncached_texts:
            logger.debug(
                "Encoding %d new texts (batch_size=%d)",
                len(uncached_texts),
                self.batch_size,
            )
            new_embeddings: np.ndarray = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=len(uncached_texts) > 50,
                convert_to_numpy=True,
            )
            for local_i, (orig_i, text) in enumerate(
                zip(uncached_indices, uncached_texts)
            ):
                emb = new_embeddings[local_i]
                results[orig_i] = emb
                self._save_cached(self._cache_key(text), emb)

        return np.vstack([r for r in results if r is not None])

    def embed_single(self, text: str) -> np.ndarray:
        """Convenience wrapper — embed one text and return a 1-D array."""
        return self.embed([text])[0]

    @property
    def embedding_dim(self) -> int:
        """Return the output dimensionality of the current model."""
        model = self.model
        if hasattr(model, "get_embedding_dimension"):
            return model.get_embedding_dimension()
        return model.get_sentence_embedding_dimension()
