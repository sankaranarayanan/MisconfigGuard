"""
RAGOrchestrator — Top-level coordinator for retrieval-augmented security analysis.

Ties together all pipeline components:

    HybridRetriever         → semantic + keyword code-chunk retrieval
    SecurityKnowledgeBase   → authoritative OWASP / CIS rule retrieval
    PromptBuilder           → structured, token-aware prompt assembly
    LocalLLMClient          → local LLM (Ollama) inference
    TTL cache               → avoids redundant retrieval for repeated queries

Output
------
Every ``analyze()`` call returns a structured dict:

    {
        "query":   str,
        "issues":  [{"title", "severity", "description",
                     "affected_resource", "recommendation",
                     "cwe", "owasp"}, …],
        "evidence": {
            "code_chunks":        [chunk_dict, …],
            "security_references": [rule_dict, …],
        },
        "analysis": str,          # raw LLM response text
        "metadata": {
            "model":   str,
            "top_k_code":     int,
            "top_k_security": int,
            "retrieval": {
                "code_count":     int,
                "security_count": int,
            },
            "cached": bool,
        },
    }

Usage
-----
    orchestrator = RAGOrchestrator.from_pipeline(pipeline)
    result = orchestrator.analyze("Check Terraform for public S3 risks")
    print(result["issues"])
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from context_builder import ContextBuilder
from prompt_builder import PromptBuilder
from hybrid_retriever import HybridRetriever, RetrievalResult
from rule_aware_retriever import RuleAwareRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cache_key(query: str, metadata_filter: Optional[Dict]) -> str:
    """Return a SHA-256 cache key for the (query, filter) pair."""
    payload = json.dumps(
        {"query": query, "filter": metadata_filter or {}},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _parse_structured_output(raw: str) -> Tuple[List[dict], str]:
    """
    Extract the issues list and summary from the LLM's raw text response.

    Strategy
    --------
    1. Try ``json.loads()`` on the whole response (ideal case).
    2. Try extracting the first ``{…}`` block with a regex.
    3. Fall back to building a single-issue dict from the raw text.

    Returns ``(issues, summary)`` where issues is a list of dicts and
    summary is a plain-text string.
    """
    # 1. Direct JSON parse
    text = raw.strip()
    try:
        data = json.loads(text)
        return _normalise_issues(data.get("issues", [])), data.get("summary", text[:200])
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Extract first JSON object from the text (LLMs sometimes add preamble)
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            data = json.loads(match.group())
            return _normalise_issues(data.get("issues", [])), data.get("summary", text[:200])
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Heuristic: look for severity keywords and construct a single issue
    issues: List[dict] = []
    severity_pattern = re.compile(
        r"\b(CRITICAL|HIGH|MEDIUM|LOW|INFO)\b", re.IGNORECASE
    )
    if severity_pattern.search(text):
        severity = severity_pattern.search(text).group(1).upper()
        issues = [{
            "title":             "Potential Security Issue",
            "severity":          severity,
            "description":       text[:500],
            "affected_resource": "",
            "recommendation":    "",
            "cwe":               "",
            "owasp":             "",
        }]

    summary = text[:300] if not issues else text[:200]
    return issues, summary


def _normalise_issues(raw_issues: List[Any]) -> List[dict]:
    """
    Ensure every issue dict contains all expected keys with default values.
    """
    required = {
        "title":             "",
        "severity":          "INFO",
        "description":       "",
        "affected_resource": "",
        "recommendation":    "",
        "cwe":               "",
        "owasp":             "",
        "rule_id":           "",
        "rule_description":  "",
    }
    result = []
    for item in (raw_issues or []):
        if not isinstance(item, dict):
            continue
        normalised = dict(required)
        normalised.update(item)
        # Normalise severity to upper-case.
        normalised["severity"] = str(normalised["severity"]).upper()
        result.append(normalised)
    return result


# ---------------------------------------------------------------------------
# RAGOrchestrator
# ---------------------------------------------------------------------------


class RAGOrchestrator:
    """
    Coordinate hybrid retrieval, security-rule lookup, prompt assembly,
    and local LLM inference to produce structured security findings.

    Parameters
    ----------
    hybrid_retriever :
        ``HybridRetriever`` connected to the code-chunk vector store.
    prompt_builder :
        ``PromptBuilder`` for assembling the LLM prompt.
    llm_client :
        ``LocalLLMClient`` for calling the Ollama API.
    security_kb : optional
        ``SecurityKnowledgeBase`` for retrieving OWASP/CIS rules.
    top_k_code :
        Number of code chunks to retrieve per query.
    top_k_security :
        Number of security rules to retrieve per query.
    cache_ttl :
        Seconds before a cached retrieval result expires.  0 disables caching.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        prompt_builder:   PromptBuilder,
        llm_client:       Any,
        security_kb:      Optional[Any] = None,
        top_k_code:       int   = 5,
        top_k_security:   int   = 3,
        cache_ttl:        int   = 300,
    ) -> None:
        self.retriever      = hybrid_retriever
        self.prompt_builder = prompt_builder
        self.context_builder = ContextBuilder(prompt_builder)
        self.llm_client     = llm_client
        self.security_kb    = security_kb
        self.top_k_code     = top_k_code
        self.top_k_security = top_k_security
        self.cache_ttl      = cache_ttl
        self.rule_aware_retriever = None
        if self.security_kb is not None:
            self.rule_aware_retriever = RuleAwareRetriever(
                code_retriever=self.retriever,
                security_kb=self.security_kb,
                embedder=self.retriever.embedder,
            )

        # TTL cache: key → (result_dict, expiry_timestamp)
        self._cache: Dict[str, Tuple[dict, float]] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pipeline(
        cls,
        pipeline:        Any,
        security_kb:     Optional[Any] = None,
        top_k_code:      int   = 5,
        top_k_security:  int   = 3,
        semantic_weight: float = 0.7,
        keyword_weight:  float = 0.3,
        cache_ttl:       int   = 300,
        max_context_tokens: int = 3000,
    ) -> "RAGOrchestrator":
        """
        Convenience factory: create an ``RAGOrchestrator`` that shares the
        embedder and vector store from an existing ``RAGPipeline``.

        Parameters
        ----------
        pipeline :
            An initialised ``RAGPipeline`` instance.
        security_kb :
            Optional pre-built ``SecurityKnowledgeBase``.
        """
        retriever = HybridRetriever(
            vector_store    = pipeline.vector_store,
            embedder        = pipeline.embedder,
            semantic_weight = semantic_weight,
            keyword_weight  = keyword_weight,
        )
        prompt_builder = PromptBuilder(max_context_tokens=max_context_tokens)
        return cls(
            hybrid_retriever = retriever,
            prompt_builder   = prompt_builder,
            llm_client       = pipeline.llm_client,
            security_kb      = security_kb,
            top_k_code       = top_k_code,
            top_k_security   = top_k_security,
            cache_ttl        = cache_ttl,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        query:           str,
        metadata_filter: Optional[Dict] = None,
        stream:          bool = False,
        top_k_code:      Optional[int] = None,
        top_k_security:  Optional[int] = None,
        intent_hint:     str = "",
    ) -> dict:
        """
        Perform retrieval-augmented security analysis for *query*.

        Parameters
        ----------
        query :
            The analysis question (e.g., "Check Terraform for public S3 risks").
        metadata_filter :
            Optional cloud/file-type filter for code retrieval.
        stream :
            If ``True``, use streaming LLM inference (still returns the full
            result dict after collection).
        top_k_code :
            Override instance-level ``top_k_code`` for this call.
        top_k_security :
            Override instance-level ``top_k_security`` for this call.

        Returns
        -------
        dict
            Structured analysis result (see module docstring for schema).
        """
        k_code = top_k_code or self.top_k_code
        k_sec  = top_k_security or self.top_k_security

        # Cache lookup
        if self.cache_ttl > 0:
            key = _cache_key(query, metadata_filter)
            if key in self._cache:
                cached_result, expiry = self._cache[key]
                if time.time() < expiry:
                    logger.debug("RAGOrchestrator cache hit for query: %.60s…", query)
                    result = dict(cached_result)
                    result["metadata"] = dict(result.get("metadata", {}))
                    result["metadata"]["cached"] = True
                    return result

        matched_resources: List[dict] = []
        if self.rule_aware_retriever is not None:
            bundle = self.rule_aware_retriever.retrieve(
                query=query,
                top_k_code=k_code,
                top_k_rules=k_sec,
                metadata_filter=metadata_filter,
            )
            code_results = bundle.get("code_results", [])
            security_results = bundle.get("security_results", [])
            matched_resources = bundle.get("matched_resources", [])
        else:
            code_results = self._retrieve_code(query, k_code, metadata_filter)
            security_results = self._retrieve_security_rules(query, k_sec, code_results)

        # Assemble prompt
        prompt = self.context_builder.build(
            query            = query,
            code_results     = code_results,
            security_results = security_results,
            matched_resources = matched_resources,
            intent_hint      = intent_hint,
        )

        # Call LLM
        raw_analysis = self._call_llm(prompt, stream=stream)

        # Parse structured output
        issues, summary = _parse_structured_output(raw_analysis)
        issues = self._enrich_issues_with_rules(issues, security_results)

        result = {
            "query":  query,
            "issues": issues,
            "evidence": {
                "code_chunks": [
                    r.to_dict() for r in code_results
                ],
                "security_references": [
                    r.to_dict() if hasattr(r, "to_dict") else r
                    for r in security_results
                ],
            },
            "analysis": raw_analysis,
            "summary":  summary,
            "metadata": {
                "model":          getattr(self.llm_client, "model", "unknown"),
                "top_k_code":     k_code,
                "top_k_security": k_sec,
                "retrieval": {
                    "code_count":     len(code_results),
                    "security_count": len(security_results),
                },
                "matched_resources": matched_resources,
                "cached": False,
            },
        }

        # Cache write
        if self.cache_ttl > 0:
            self._cache[key] = (result, time.time() + self.cache_ttl)

        return result

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Invalidate all cached retrieval results."""
        self._cache.clear()
        logger.debug("RAGOrchestrator cache cleared.")

    def cache_stats(self) -> Dict[str, int]:
        """Return current cache state: total entries and live (non-expired) count."""
        now   = time.time()
        total = len(self._cache)
        live  = sum(1 for _, exp in self._cache.values() if now < exp)
        return {"total": total, "live": live, "expired": total - live}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve_code(
        self,
        query:           str,
        top_k:           int,
        metadata_filter: Optional[Dict],
    ) -> List[RetrievalResult]:
        """Retrieve code chunks using hybrid retrieval."""
        try:
            return self.retriever.retrieve(
                query           = query,
                top_k           = top_k,
                metadata_filter = metadata_filter,
            )
        except Exception as exc:
            logger.warning("Code retrieval failed: %s", exc)
            return []

    def _retrieve_security_rules(
        self,
        query:        str,
        top_k:        int,
        code_results: List[RetrievalResult],
    ) -> list:
        """Retrieve security rules from the knowledge base."""
        if self.security_kb is None or self.security_kb.total_rules == 0:
            return []
        try:
            query_emb = self.retriever._embed_query(query)
            return self.security_kb.search(query_emb, top_k=top_k)
        except Exception as exc:
            logger.warning("Security KB retrieval failed: %s", exc)
            return []

    def _enrich_issues_with_rules(self, issues: List[dict], security_results: List[Any]) -> List[dict]:
        if not issues or not security_results:
            return issues
        primary_rule = security_results[0]
        for issue in issues:
            if not issue.get("rule_id"):
                issue["rule_id"] = getattr(primary_rule, "rule_id", "")
            if not issue.get("rule_description"):
                issue["rule_description"] = getattr(primary_rule, "description", "")
        return issues

    def _call_llm(self, prompt: str, stream: bool = False) -> str:
        """Call the local LLM and return the raw response text."""
        try:
            if stream and hasattr(self.llm_client, "stream_generate"):
                chunks = []
                for tok in self.llm_client.stream_generate(prompt):
                    print(tok, end="", flush=True)
                    chunks.append(tok)
                print()
                return "".join(chunks)
            return self.llm_client.generate(prompt)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return f'{{"issues": [], "summary": "LLM unavailable: {exc}"}}'
