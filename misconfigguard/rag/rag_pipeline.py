"""
RAGPipeline — End-to-end orchestrator that wires together all components:

    FileScanner  ──►  FileParser  ──►  Chunker / IntelligentChunker
                                                  │
                                                  ▼
                                       EmbeddingGenerator
                                                  │
                                                  ▼
                                       VectorStoreManager  ◄──  persist / load
                                                  │
                                    query ──►  similarity search
                                                  │
                                        [dependency expansion]
                                                  │
                                                  ▼
                                       LocalLLMClient  ──►  analysis report

Two ingestion entry points:
    ingest_directory(path)           — local directory scan
    ingest_repository(url, token)    — remote Git clone

One query entry point:
    analyze(query, top_k)            — RAG retrieval + LLM analysis

Performance features:
    • Parallel file parsing via ThreadPoolExecutor (I/O-bound work).
    • Incremental indexing: already-indexed files are skipped using a
      SHA-256 content hash registry persisted to disk.
    • Intelligent chunking: semantic / dependency-aware chunking via
      IntelligentChunker when configured (falls back to Chunker).
    • Dependency expansion: retrieved chunks can be enriched with their
      related blocks at query time.
"""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from chunker import Chunk, Chunker
from embedding_generator import EmbeddingGenerator
from file_parser import FileParser
from file_scanner import FileScanner
from git_ingestor import GitIngestor
from intelligent_chunker import IntelligentChunker
from local_llm_client import LocalLLMClient
from resource_tagger import ResourceTagger
from vector_store_manager import SearchResult, VectorStoreManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Incremental-index registry
# ---------------------------------------------------------------------------

class _IndexRegistry:
    """
    Persists a {file_path → content_sha256} mapping so that unchanged
    files are skipped on subsequent ingestion runs.
    """

    def __init__(self, registry_path: str = "./cache/index_registry.json"):
        self._path = Path(registry_path)
        self._registry: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._registry, indent=2))

    def has_entries(self) -> bool:
        return bool(self._registry)

    def clear(self) -> None:
        self._registry.clear()
        try:
            if self._path.exists():
                self._path.unlink()
        except OSError:
            logger.warning("Failed to remove stale index registry: %s", self._path)

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def is_indexed(self, file_path: str, content: str) -> bool:
        """Return True if this exact content was indexed previously."""
        return self._registry.get(file_path) == self._hash(content)

    def mark_indexed(self, file_path: str, content: str) -> None:
        self._registry[file_path] = self._hash(content)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    High-level orchestrator for the MisconfigGuard RAG pipeline.

    All components are injected through the constructor so that each
    can be independently tested, swapped, or reconfigured.
    """

    def __init__(
        self,
        scanner: Optional[FileScanner] = None,
        parser: Optional[FileParser] = None,
        chunker=None,
        embedder: Optional[EmbeddingGenerator] = None,
        rerank_embedder: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStoreManager] = None,
        llm_client: Optional[LocalLLMClient] = None,
        batch_embed_size: int = 64,
        max_workers: int = 8,
        incremental: bool = True,
        registry_path: str = "./cache/index_registry.json",
        expand_dependencies: bool = False,
        query_routing_cfg: Optional[dict] = None,
        retrieval_cfg: Optional[dict] = None,
        iam_cfg: Optional[dict] = None,
        workload_identity_cfg: Optional[dict] = None,
        prompt_injection_cfg: Optional[dict] = None,
        secrets_cfg: Optional[dict] = None,
    ):
        """
        Args:
            scanner:             FileScanner for directory traversal.
            parser:              FileParser for normalisation.
            chunker:             Chunker or IntelligentChunker for text splitting.
                                 Defaults to IntelligentChunker (semantic strategy).
            embedder:            EmbeddingGenerator for dense vectors.
            vector_store:        VectorStoreManager (FAISS or Chroma).
            llm_client:          LocalLLMClient for Ollama interaction.
            batch_embed_size:    Number of chunks embedded per forward pass.
            max_workers:         ThreadPoolExecutor workers for parallel parsing.
            incremental:         Skip already-indexed files (content-hash based).
            registry_path:       Path to the incremental index registry file.
            expand_dependencies: When True and the chunker is an IntelligentChunker,
                                 retrieved results are enriched with their dependency
                                 chunks before building the LLM context.
        """
        self.scanner = scanner or FileScanner()
        self.parser = parser or FileParser()
        self.chunker = chunker or IntelligentChunker()
        self.embedder = embedder or EmbeddingGenerator()
        self.rerank_embedder = rerank_embedder
        self.vector_store = vector_store or VectorStoreManager()
        self.llm_client = llm_client or LocalLLMClient()
        self.resource_tagger = ResourceTagger()
        self.batch_embed_size = batch_embed_size
        self.max_workers = max_workers
        self.incremental = incremental
        self.expand_dependencies = expand_dependencies
        self._registry = _IndexRegistry(registry_path) if incremental else None
        self.query_routing_cfg = {
            "enabled": True,
            "use_llm_routing": True,
            "routing_model": getattr(self.llm_client, "model", None),
            "routing_max_tokens": 20,
            "routing_cache_ttl": 300,
            "log_intent": True,
            **(query_routing_cfg or {}),
        }
        self.retrieval_cfg = retrieval_cfg or {}
        self.iam_cfg = iam_cfg or {}
        self.workload_identity_cfg = workload_identity_cfg or {}
        self.prompt_injection_cfg = prompt_injection_cfg or {}
        self.secrets_cfg = secrets_cfg or {}

    # ==================================================================
    # Shared ingestion kernel
    # ==================================================================

    def _process_batch(self, chunk_batch: list) -> None:
        """Embed *chunk_batch* and push vectors + metadata to the store."""
        tagged_chunks = [self.resource_tagger.tag_chunk(c.to_dict()) for c in chunk_batch]
        texts = [chunk["text"] for chunk in tagged_chunks]
        embeddings = self.embedder.embed(texts)
        self.vector_store.add(embeddings, tagged_chunks)

    def _reconcile_incremental_state(self) -> None:
        """Load existing persistence or reset stale incremental state."""
        if not self.incremental or self._registry is None:
            return
        if self.vector_store.total_vectors > 0:
            return
        if self.vector_store.load(expected_dim=self.embedder.embedding_dim):
            return
        if self._registry.has_entries() or self.vector_store.has_persisted_state():
            logger.warning(
                "Incremental registry/vector store state is stale; clearing cached state and rebuilding index"
            )
            self._registry.clear()
            self.vector_store.reset_persistence()

    def _should_skip(self, record: dict) -> bool:
        """Return True if this file was already indexed with identical content."""
        if not self.incremental or self._registry is None:
            return False
        return self._registry.is_indexed(record["file_path"], record["content"])

    def _mark_indexed(self, record: dict) -> None:
        if self.incremental and self._registry is not None:
            self._registry.mark_indexed(record["file_path"], record["content"])

    def _ingest_records(self, records) -> int:
        """
        Stream FileRecords → filter incremental → Chunks → embeddings → store.

        Returns the total number of NEW chunks indexed (skipped files excluded).
        """
        buffer: list = []
        total = 0
        skipped = 0

        for record in records:
            if self._should_skip(record):
                skipped += 1
                continue

            for chunk in self.chunker.chunk_record(record):
                buffer.append(chunk)
                if len(buffer) >= self.batch_embed_size:
                    self._process_batch(buffer)
                    total += len(buffer)
                    logger.debug("Indexed %d chunks so far...", total)
                    buffer.clear()

            self._mark_indexed(record)

        # Flush the remaining partial batch
        if buffer:
            self._process_batch(buffer)
            total += len(buffer)

        # Persist incremental registry + vector index
        if self._registry is not None:
            self._registry.save()
        self.vector_store.save()

        if skipped:
            logger.info("Skipped %d already-indexed file(s) (incremental mode)", skipped)

        return total

    def _parse_file_task(self, file_path, metadata=None):
        """Worker function for parallel file parsing."""
        return self.parser.parse_file(file_path, metadata=metadata)

    def _ingest_directory_parallel(self, directory: str) -> int:
        """
        Parse files in parallel using ThreadPoolExecutor (I/O-bound),
        then stream chunks through the sequential embedding pipeline.
        """
        from pathlib import Path  # noqa: F811 – ensure available in this scope

        file_paths = list(self.scanner.scan(directory))
        logger.info(
            "Found %d supported files in %s — parsing with %d workers",
            len(file_paths),
            directory,
            self.max_workers,
        )

        records = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self._parse_file_task, fp): fp
                for fp in file_paths
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    record = future.result()
                    if record is not None:
                        records.append(record)
                except Exception as exc:
                    logger.error("Error parsing %s: %s", path, exc)

        return self._ingest_records(records)

    # ==================================================================
    # Ingestion entry points
    # ==================================================================

    def ingest_directory(self, directory: str, parallel: bool = True) -> int:
        """
        Recursively ingest all supported files under *directory*.

        Args:
            directory: Path to scan.
            parallel:  Parse files in parallel (ThreadPoolExecutor).
                       Disable for simpler single-threaded debugging.

        Returns:
            Total number of NEW chunks added to the vector store.
        """
        logger.info("Ingesting local directory: %s", directory)
        self._reconcile_incremental_state()
        if parallel:
            total = self._ingest_directory_parallel(directory)
        else:
            records = self.parser.parse_directory(directory, self.scanner)
            total = self._ingest_records(records)
        logger.info(
            "Directory ingestion complete — %d new chunks indexed from %s",
            total,
            directory,
        )
        return total

    def ingest_repository(
        self,
        url: str,
        token: Optional[str] = None,
        branch: Optional[str] = None,
        clone_dir: str = "./tmp/repos",
    ) -> int:
        """
        Clone *url* and ingest all supported files.

        Args:
            url:       Git repository URL (HTTPS or SSH).
            token:     Personal access token for private repos.
            branch:    Branch to check out (default: remote HEAD).
            clone_dir: Local directory for cloned repos.

        Returns:
            Total number of chunks added to the vector store.
        """
        logger.info("Ingesting Git repository: %s", url)
        self._reconcile_incremental_state()
        ingestor = GitIngestor(clone_dir=clone_dir, scanner=self.scanner)
        records = self.parser.parse_repository(
            ingestor, url, token=token, branch=branch
        )
        total = self._ingest_records(records)
        logger.info(
            "Repository ingestion complete — %d chunks indexed from %s",
            total,
            url,
        )
        return total

    # ==================================================================
    # Retrieval & analysis
    # ==================================================================

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Embed *query* and return the *top_k* most relevant chunks.

        When ``expand_dependencies=True`` and the chunker is an
        ``IntelligentChunker``, dependency chunks are appended (deduplicated)
        to enrich the context returned to the LLM.
        """
        query_vec = self.embedder.embed_single(query)
        results = self.vector_store.search(query_vec, top_k=top_k)

        if self.expand_dependencies and isinstance(self.chunker, IntelligentChunker):
            results = self._expand_with_dependencies(results)

        return results

    def _expand_with_dependencies(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Append dependency chunks to *results* (no duplicates, rank preserved)."""
        seen_ids: set = {
            r.chunk.get("chunk_id", r.chunk.get("file_path", ""))
            for r in results
        }
        extra: List[SearchResult] = []
        resolver = self.chunker.resolver  # type: ignore[attr-defined]

        for result in results:
            chunk_id = result.chunk.get("chunk_id")
            if not chunk_id:
                continue
            for dep_chunk in resolver.expand_dependencies([chunk_id], max_depth=1):
                dep_id = dep_chunk.chunk_id
                if dep_id not in seen_ids:
                    seen_ids.add(dep_id)
                    # Dependency chunks are ranked after the primary results.
                    extra.append(
                        SearchResult(
                            rank=len(results) + len(extra) + 1,
                            score=0.0,
                            chunk=dep_chunk.to_dict(),
                        )
                    )

        return results + extra

    @staticmethod
    def _build_context(results: List[SearchResult]) -> str:
        """Format retrieved chunks into a readable LLM context block."""
        parts = []
        for r in results:
            file_type      = r.chunk.get("file_type", "")
            chunk_id       = r.chunk.get("chunk_id", "")
            chunk_index    = r.chunk.get("chunk_index", 0)
            cloud_provider = r.chunk.get("cloud_provider", "")
            resource_type  = r.chunk.get("resource_type", "")
            score_str      = f", similarity={r.score:.3f}" if r.score > 0 else ""

            # Build an optional context annotation line.
            annotations = []
            if cloud_provider and cloud_provider != "unknown":
                annotations.append(f"cloud={cloud_provider}")
            if resource_type:
                annotations.append(f"resource={resource_type}")
            if chunk_id:
                annotations.append(f"id={chunk_id}")
            ann_str = f"  [{', '.join(annotations)}]" if annotations else ""

            parts.append(
                f"### [{r.rank}] {r.chunk.get('file_path', '')}  "
                f"(chunk #{chunk_index}{score_str}){ann_str}\n"
                f"```{file_type}\n{r.chunk.get('text', '')}\n```"
            )
        return "\n\n".join(parts)

    def analyze(
        self,
        query: str,
        top_k: int = 5,
        stream: bool = False,
    ) -> dict:
        """
        Full RAG query: retrieve relevant chunks then ask the LLM.

        Args:
            query:  Security question or analysis directive.
            top_k:  Number of chunks to retrieve.
            stream: Stream LLM tokens to stdout while generating.

        Returns:
            {
                "query":    str,
                "context":  str,          # formatted retrieved chunks
                "results":  list[dict],   # raw chunk dicts
                "analysis": str,          # LLM response
            }
        """
        logger.info("RAG query: %r", query)

        results = self.retrieve(query, top_k=top_k)
        if not results:
            return {
                "query": query,
                "context": "",
                "results": [],
                "analysis": (
                    "No relevant content found in the vector store. "
                    "Run ingest_directory() or ingest_repository() first."
                ),
            }

        if not self.llm_client.is_available():
            logger.warning(
                "Ollama is not reachable - returning retrieval results only. "
                "Start Ollama with: ollama serve"
            )
            return {
                "query": query,
                "context": self._build_context(results),
                "results": [r.chunk for r in results],
                "analysis": (
                    "[Ollama unavailable] Install Ollama from https://ollama.com and run: "
                    f"ollama pull {self.llm_client.model} && ollama serve"
                ),
            }

        routed = self.query(
            query=query,
            top_k=top_k,
            use_llm_routing=self.query_routing_cfg.get("use_llm_routing", True),
            structured=False,
            stream=stream,
        )
        return {
            "query": query,
            "context": routed.get("context", ""),
            "results": routed.get("results", []),
            "analysis": routed.get("analysis", ""),
        }

    def query(
        self,
        query: str,
        top_k: int = 5,
        use_llm_routing: bool = True,
        structured: bool = False,
        stream: bool = False,
    ) -> dict:
        from query_dispatcher import QueryDispatcher
        from query_router import QueryRouter

        router = QueryRouter(
            llm_client=self.llm_client,
            use_llm_routing=use_llm_routing and self.query_routing_cfg.get("enabled", True),
            routing_model=self.query_routing_cfg.get("routing_model"),
            routing_max_tokens=self.query_routing_cfg.get("routing_max_tokens", 20),
            cache_ttl=self.query_routing_cfg.get("routing_cache_ttl", 300),
        )
        intent = router.classify(query)
        if self.query_routing_cfg.get("log_intent", True):
            logger.info("[QueryRouter] intent=%s for query: %r", intent.value, query)

        dispatcher = QueryDispatcher(rag_pipeline=self)
        result = dispatcher.dispatch(query, intent, top_k=top_k, stream=stream)
        if structured and not result.get("findings"):
            findings, summary = [], ""
            try:
                from rag_orchestrator import _parse_structured_output
                findings, summary = _parse_structured_output(result.get("analysis", ""))
            except Exception:
                findings, summary = [], ""
            if findings:
                result["findings"] = findings
                result["issues"] = findings
            if summary and not result.get("summary"):
                result["summary"] = summary
        return result

    def analyze_structured(
        self,
        query:           str,
        top_k_code:      int = 5,
        top_k_security:  int = 3,
        metadata_filter: Optional[dict] = None,
        stream:          bool = False,
        security_kb=None,
        semantic_weight: float = 0.7,
        keyword_weight:  float = 0.3,
        cache_ttl:       int = 300,
    ) -> dict:
        """
        Structured security analysis using HybridRetriever + SecurityKnowledgeBase.

        Returns the richer output dict produced by ``RAGOrchestrator.analyze()``:

            {
                "query":    str,
                "issues":   [{"title", "severity", "description", ...}, ...],
                "evidence": {"code_chunks": [...], "security_references": [...]},
                "analysis": str,          # raw LLM text
                "summary":  str,
                "metadata": {...},
            }

        Args:
            query:           Security question or analysis directive.
            top_k_code:      Number of code chunks to retrieve.
            top_k_security:  Number of security rules to retrieve.
            metadata_filter: Optional cloud / file-type filter dict.
            stream:          Stream LLM tokens (response still returned fully).
            security_kb:     Optional ``SecurityKnowledgeBase`` instance.
            semantic_weight: Weight for semantic similarity (default 0.7).
            keyword_weight:  Weight for keyword relevance (default 0.3).
            cache_ttl:       Retrieval cache TTL in seconds.
        """
        from rag_orchestrator import RAGOrchestrator

        retrieval_cfg = getattr(self, "retrieval_cfg", {}) or {}
        orchestrator = RAGOrchestrator.from_pipeline(
            pipeline            = self,
            security_kb         = security_kb,
            top_k_code          = top_k_code,
            top_k_security      = top_k_security,
            semantic_weight     = semantic_weight,
            keyword_weight      = keyword_weight,
            cache_ttl           = cache_ttl,
            # FIX: wire max_context_tokens from config instead of hardcoded 3000
            max_context_tokens  = retrieval_cfg.get("max_context_tokens", 8000),
        )
        return orchestrator.analyze(
            query           = query,
            metadata_filter = metadata_filter,
            stream          = stream,
        )

    # ==================================================================
    # Index lifecycle helpers
    # ==================================================================

    def load_index(self) -> bool:
        """
        Load a previously saved vector index from disk.

        Returns True if a saved index was found and loaded.
        """
        loaded = self.vector_store.load()
        if loaded:
            logger.info(
                "Vector index loaded (%d vectors)",
                self.vector_store.total_vectors,
            )
        else:
            logger.info("No saved vector index found")
        return loaded

    @property
    def total_indexed(self) -> int:
        """Total number of chunk vectors currently in the store."""
        return self.vector_store.total_vectors
