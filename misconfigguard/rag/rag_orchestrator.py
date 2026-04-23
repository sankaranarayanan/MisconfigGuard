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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from context_builder import ContextBuilder
from prompt_builder import PromptBuilder
from hybrid_retriever import HybridRetriever, RetrievalResult
from rule_aware_retriever import RuleAwareRetriever

logger = logging.getLogger(__name__)


@dataclass
class FailureSignals:
    """Quality signals used by the Skill-RAG router."""

    hallucination_score: float
    grounding_score: float
    confidence_score: float
    is_failure: bool
    reason: str


@dataclass
class IterationRecord:
    """Single Skill-RAG iteration trace record."""

    query: str
    docs: List[str]
    answer: str
    failure_signals: FailureSignals
    chosen_skill: str


@dataclass
class SkillRAGResult:
    """Final Skill-RAG output with deterministic trace."""

    final_answer: str
    iterations_trace: List[IterationRecord]


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


def _tokenize(text: str) -> List[str]:
    """Return lowercase alphanumeric tokens for simple overlap scoring."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


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

    # 3. Fallback: log the failure and return an empty issues list.
    # Creating synthetic issues from heuristic severity-keyword scanning produces
    # ghost findings with no affected_resource/CWE — a primary hallucination source.
    logger.warning(
        "LLM returned non-JSON output; no issues synthesised. Raw (first 300 chars): %s",
        text[:300],
    )
    return [], text[:300]


_VALID_SEVERITIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}
_CWE_PATTERN  = re.compile(r"^CWE-\d+$")
_OWASP_PATTERN = re.compile(r"^OWASP A\d{2}:\d{4}")


def _validate_issue(issue: dict) -> Tuple[bool, List[str]]:
    """
    Validate a single normalised issue dict against schema rules.

    Returns ``(is_valid, [error_messages])``.  Invalid CRITICAL/HIGH issues are
    logged as warnings so analysts can triage model output quality.
    """
    errors: List[str] = []
    sev = issue.get("severity", "")
    if sev not in _VALID_SEVERITIES:
        errors.append(f"invalid severity '{sev}'")
    if sev in {"CRITICAL", "HIGH"}:
        if not issue.get("affected_resource"):
            errors.append("CRITICAL/HIGH requires non-empty affected_resource")
        if not issue.get("cwe"):
            errors.append("CRITICAL/HIGH requires a CWE mapping")
        if not issue.get("evidence_snippet"):
            errors.append("CRITICAL/HIGH requires an evidence_snippet")
    cwe = issue.get("cwe", "")
    if cwe and not _CWE_PATTERN.match(cwe):
        errors.append(f"invalid CWE format '{cwe}' — expected 'CWE-NNN'")
    owasp = issue.get("owasp", "")
    if owasp and not _OWASP_PATTERN.match(owasp):
        errors.append(f"invalid OWASP format '{owasp}' — expected 'OWASP AXX:YYYY'")
    return len(errors) == 0, errors


def _normalise_issues(raw_issues: List[Any]) -> List[dict]:
    """
    Ensure every issue dict contains all expected keys with default values,
    then validate each issue and log warnings for schema violations.
    """
    required = {
        "title":            "",
        "severity":         "INFO",
        "description":      "",
        "affected_resource":"",
        "recommendation":   "",
        "cwe":              "",
        "owasp":            "",
        "rule_id":          "",
        "rule_description": "",
        "file_path":        "",
        "evidence_snippet": "",
    }
    result = []
    for item in (raw_issues or []):
        if not isinstance(item, dict):
            continue
        normalised = dict(required)
        normalised.update(item)
        normalised["severity"] = str(normalised["severity"]).upper()
        is_valid, errors = _validate_issue(normalised)
        if not is_valid:
            logger.warning(
                "Issue schema violation in '%s' [%s]: %s",
                normalised.get("title", "<untitled>"),
                normalised["severity"],
                "; ".join(errors),
            )
        result.append(normalised)
    return result


def _compute_retrieval_confidence(
    code_results: List[Any],
    security_results: List[Any],
    top_k_code: int,
    top_k_security: int,
) -> float:
    """
    Return a [0, 1] confidence score for the retrieval quality of this query.

    Score components:
    - 0.5 × chunk coverage  (how many of top_k_code slots were filled)
    - 0.3 × avg score of top-3 chunks (proxy for semantic relevance)
    - 0.2 × rule coverage   (how many of top_k_security slots were filled)

    A score < 0.4 indicates weak retrieval and the result should be flagged
    for manual review.
    """
    chunk_coverage = min(len(code_results) / max(top_k_code, 1), 1.0)
    rule_coverage  = min(len(security_results) / max(top_k_security, 1), 1.0)

    top3_scores: List[float] = []
    for r in (code_results or [])[:3]:
        score = getattr(r, "final_score", None) or (r.get("final_score") if isinstance(r, dict) else None)
        if score is not None:
            top3_scores.append(float(score))
    avg_top3 = sum(top3_scores) / len(top3_scores) if top3_scores else 0.0

    confidence = chunk_coverage * 0.5 + avg_top3 * 0.3 + rule_coverage * 0.2
    return round(min(confidence, 1.0), 3)


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
    # Skill-RAG public API
    # ------------------------------------------------------------------

    def analyze_with_skills(
        self,
        query: str,
        metadata_filter: Optional[Dict] = None,
        stream: bool = False,
        top_k_code: Optional[int] = None,
        top_k_security: Optional[int] = None,
        intent_hint: str = "",
        max_iterations: int = 3,
    ) -> SkillRAGResult:
        """
        Run a Skill-RAG inspired iterative loop.

        Loop per iteration:
        retrieve -> generate -> detect failure -> route skill -> retry
        """
        k_code = top_k_code or self.top_k_code
        k_sec = top_k_security or self.top_k_security

        current_query = query
        iterations_trace: List[IterationRecord] = []
        final_answer = ""

        for _ in range(max_iterations):
            matched_resources: List[dict] = []
            if self.rule_aware_retriever is not None:
                bundle = self.rule_aware_retriever.retrieve(
                    query=current_query,
                    top_k_code=k_code,
                    top_k_rules=k_sec,
                    metadata_filter=metadata_filter,
                )
                code_results = bundle.get("code_results", [])
                security_results = bundle.get("security_results", [])
                matched_resources = bundle.get("matched_resources", [])
            else:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    code_future = executor.submit(self._retrieve_code, current_query, k_code, metadata_filter)
                    security_future = executor.submit(self._retrieve_security_rules, current_query, k_sec, [])
                    code_results = code_future.result()
                    security_results = security_future.result()

            prompt = self.context_builder.build(
                query=current_query,
                code_results=code_results,
                security_results=security_results,
                matched_resources=matched_resources,
                intent_hint=intent_hint,
            )

            answer = self._call_llm(prompt, stream=stream)
            final_answer = answer

            docs = self._extract_doc_texts(code_results, security_results)
            signals = self.detect_failure_signals(current_query, answer, docs)
            chosen_skill = self.route_skill(current_query, signals)

            iterations_trace.append(
                IterationRecord(
                    query=current_query,
                    docs=docs,
                    answer=answer,
                    failure_signals=signals,
                    chosen_skill=chosen_skill,
                )
            )

            if chosen_skill == "exit" or not signals.is_failure:
                break

            if chosen_skill == "rewrite":
                current_query = self.rewrite_query(current_query)
            elif chosen_skill == "decompose":
                subqueries = self.decompose_query(current_query)
                current_query = " ; ".join(subqueries)
            elif chosen_skill == "focus":
                current_query = self.focus_query(current_query, docs)

        return SkillRAGResult(final_answer=final_answer, iterations_trace=iterations_trace)

    def detect_failure_signals(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[str],
    ) -> FailureSignals:
        """Compute concrete failure scores and rule-based failure status."""
        answer_tokens = _tokenize(answer)
        doc_tokens = set(_tokenize(" ".join(retrieved_docs)))

        if not answer_tokens:
            hallucination_score = 1.0
            grounding_score = 0.0
        else:
            missing = [tok for tok in answer_tokens if tok not in doc_tokens]
            overlap = [tok for tok in answer_tokens if tok in doc_tokens]
            hallucination_score = len(missing) / len(answer_tokens)
            grounding_score = len(overlap) / len(answer_tokens)

        confidence_score = 1.0
        answer_lower = (answer or "").lower()
        if "i think" in answer_lower:
            confidence_score -= 0.3
        if "maybe" in answer_lower:
            confidence_score -= 0.2
        if "possibly" in answer_lower:
            confidence_score -= 0.2
        if len((answer or "").split()) < 5:
            confidence_score -= 0.4
        confidence_score = max(0.0, min(1.0, confidence_score))

        reasons: List[str] = []
        if hallucination_score > 0.5:
            reasons.append("hallucination_score > 0.5")
        if grounding_score < 0.3:
            reasons.append("grounding_score < 0.3")
        if confidence_score < 0.5:
            reasons.append("confidence_score < 0.5")

        return FailureSignals(
            hallucination_score=round(hallucination_score, 3),
            grounding_score=round(grounding_score, 3),
            confidence_score=round(confidence_score, 3),
            is_failure=bool(reasons),
            reason="; ".join(reasons) if reasons else "quality acceptable",
        )

    def route_skill(self, query: str, signals: FailureSignals) -> str:
        """Deterministic router with fixed rule priority."""
        query_lower = (query or "").lower()

        if signals.hallucination_score > 0.5:
            return "rewrite"
        if signals.grounding_score < 0.3:
            return "focus"
        if self._has_multiple_intents(query_lower):
            return "decompose"
        return "exit"

    def rewrite_query(self, query: str) -> str:
        """Simplify query by removing common filler terms."""
        filler = {
            "please",
            "kindly",
            "just",
            "actually",
            "basically",
            "i",
            "think",
            "maybe",
            "possibly",
            "can",
            "could",
            "you",
            "help",
            "me",
        }
        tokens = [tok for tok in _tokenize(query) if tok not in filler]
        return " ".join(tokens) if tokens else query

    def decompose_query(self, query: str) -> List[str]:
        """Split a multi-intent query into sub-queries using and/or separators."""
        parts = re.split(r"\b(?:and|or)\b", query, flags=re.IGNORECASE)
        subqueries = [part.strip(" ,;:.\t\n") for part in parts if part.strip(" ,;:.\t\n")]
        return subqueries or [query]

    def focus_query(self, query: str, context: List[str]) -> str:
        """Append top frequent context keywords to focus follow-up retrieval."""
        stopwords = {
            "the",
            "is",
            "a",
            "an",
            "to",
            "of",
            "for",
            "in",
            "on",
            "with",
            "and",
            "or",
            "this",
            "that",
            "it",
            "as",
            "by",
            "be",
            "are",
            "at",
            "from",
            "check",
            "issue",
        }
        frequencies: Dict[str, int] = {}
        for token in _tokenize(" ".join(context)):
            if token in stopwords or len(token) <= 2:
                continue
            frequencies[token] = frequencies.get(token, 0) + 1

        ranked = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
        keywords = [token for token, _ in ranked[:5]]
        if not keywords:
            return query
        return f"{query} focus:{' '.join(keywords)}"

    def _has_multiple_intents(self, query_lower: str) -> bool:
        return (
            " and " in f" {query_lower} "
            or " or " in f" {query_lower} "
            or "," in query_lower
            or ";" in query_lower
        )

    def _extract_doc_texts(self, code_results: List[Any], security_results: List[Any]) -> List[str]:
        """Flatten retrieved artifacts into plain text evidence for scoring."""
        docs: List[str] = []
        for result in code_results or []:
            if hasattr(result, "chunk"):
                chunk = getattr(result, "chunk", {})
                if isinstance(chunk, dict):
                    docs.append(str(chunk.get("text") or chunk.get("content") or ""))
            elif isinstance(result, dict):
                docs.append(str(result.get("text") or result.get("content") or ""))
            elif isinstance(result, str):
                docs.append(result)

        for result in security_results or []:
            if hasattr(result, "description"):
                docs.append(str(getattr(result, "description", "")))
            elif isinstance(result, dict):
                docs.append(str(result.get("description") or result.get("text") or ""))

        return [doc for doc in docs if doc]

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
            rerank_embedder = getattr(pipeline, "rerank_embedder", None),
            semantic_weight = semantic_weight,
            keyword_weight  = keyword_weight,
            rerank_top_k    = pipeline.retrieval_cfg.get("rerank_top_k"),
            max_workers     = pipeline.retrieval_cfg.get("parallel_workers", 8),
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
            with ThreadPoolExecutor(max_workers=2) as executor:
                code_future = executor.submit(self._retrieve_code, query, k_code, metadata_filter)
                security_future = executor.submit(self._retrieve_security_rules, query, k_sec, [])
                code_results = code_future.result()
                security_results = security_future.result()

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
