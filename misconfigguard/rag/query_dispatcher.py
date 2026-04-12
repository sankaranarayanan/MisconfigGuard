from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from context_builder import ContextBuilder
from hybrid_retriever import HybridRetriever
from prompt_builder import PromptBuilder
from query_router import QueryIntent
from rag_orchestrator import RAGOrchestrator, _parse_structured_output
from security_kb import SecurityKnowledgeBase


class QueryDispatcher:
    """Route classified queries to the appropriate analyzer or RAG path."""

    def __init__(
        self,
        rag_pipeline,
        iam_analyzer=None,
        workload_identity_analyzer=None,
        prompt_injection_analyzer=None,
        rule_aware_retriever=None,
        context_builder=None,
    ):
        from iam_analyzer import IAMSecurityAnalyzer
        from prompt_injection_analyzer import PromptInjectionAnalyzer
        from secrets_analyzer import HardcodedSecretsAnalyzer
        from workload_identity_analyzer import WorkloadIdentitySecurityAnalyzer

        self.rag_pipeline = rag_pipeline
        self.context_builder = context_builder or ContextBuilder(PromptBuilder())
        self._retrieval_cfg = getattr(rag_pipeline, "retrieval_cfg", {}) or {}
        self._query_routing_cfg = getattr(rag_pipeline, "query_routing_cfg", {}) or {}
        self._hybrid_retriever = None
        has_retrieval_components = hasattr(rag_pipeline, "vector_store") and hasattr(rag_pipeline, "embedder")
        if has_retrieval_components:
            self._hybrid_retriever = HybridRetriever(
                vector_store=rag_pipeline.vector_store,
                embedder=rag_pipeline.embedder,
                rerank_embedder=getattr(rag_pipeline, "rerank_embedder", None),
                semantic_weight=self._retrieval_cfg.get("semantic_weight", 0.7),
                keyword_weight=self._retrieval_cfg.get("keyword_weight", 0.3),
                rerank_top_k=self._retrieval_cfg.get("rerank_top_k"),
            )
        self._security_kb = None
        self.rule_aware_retriever = rule_aware_retriever

        self.iam_analyzer = iam_analyzer
        self.workload_identity_analyzer = workload_identity_analyzer
        self.secrets_analyzer = None
        self.prompt_injection_analyzer = prompt_injection_analyzer

        if has_retrieval_components:
            self.iam_analyzer = self.iam_analyzer or IAMSecurityAnalyzer(
                pipeline=rag_pipeline,
                use_llm=True,
                max_roles_per_identity=(getattr(rag_pipeline, "iam_cfg", {}) or {}).get("max_roles_per_identity", 3),
                top_k_code=(getattr(rag_pipeline, "iam_cfg", {}) or {}).get("top_k_code", 5),
                top_k_security=(getattr(rag_pipeline, "iam_cfg", {}) or {}).get("top_k_security", 3),
            )
            self.workload_identity_analyzer = self.workload_identity_analyzer or WorkloadIdentitySecurityAnalyzer(
                pipeline=rag_pipeline,
                use_llm=True,
                top_k_code=(getattr(rag_pipeline, "workload_identity_cfg", {}) or {}).get("top_k_code", 5),
                top_k_security=(getattr(rag_pipeline, "workload_identity_cfg", {}) or {}).get("top_k_security", 3),
            )
            self.secrets_analyzer = HardcodedSecretsAnalyzer(
                pipeline=rag_pipeline,
                use_llm=True,
                top_k_code=(getattr(rag_pipeline, "secrets_cfg", {}) or {}).get("top_k_code", 5),
                top_k_security=(getattr(rag_pipeline, "secrets_cfg", {}) or {}).get("top_k_security", 3),
                entropy_threshold=(getattr(rag_pipeline, "secrets_cfg", {}) or {}).get("entropy_threshold", 4.5),
            )
            self.prompt_injection_analyzer = self.prompt_injection_analyzer or PromptInjectionAnalyzer(
                pipeline=rag_pipeline,
                use_llm=True,
                top_k_security=(getattr(rag_pipeline, "prompt_injection_cfg", {}) or {}).get("top_k_security", 3),
            )

    def dispatch(self, query: str, intent: QueryIntent, top_k: int = 5, stream: bool = False) -> dict:
        handlers = {
            QueryIntent.IAM: self._handle_iam,
            QueryIntent.WORKLOAD_IDENTITY: self._handle_workload_identity,
            QueryIntent.SECRETS: self._handle_secrets,
            QueryIntent.PROMPT_INJECTION: self._handle_prompt_injection,
            QueryIntent.NETWORK: self._handle_network,
            QueryIntent.COMPLIANCE: self._handle_compliance,
            QueryIntent.GENERAL_SECURITY: self._handle_general,
        }
        result = handlers.get(intent, self._handle_general)(query, top_k, stream=stream)
        result["intent"] = intent.value
        result["query"] = query
        result.setdefault("issues", result.get("findings", []))
        return result

    def _load_security_kb(self):
        if self._security_kb is None:
            self._security_kb = SecurityKnowledgeBase(embedder=self.rag_pipeline.embedder)
            self._security_kb.load_or_build()
        return self._security_kb

    def _build_orchestrator(self, top_k: int) -> RAGOrchestrator:
        security_kb = self._load_security_kb()
        orchestrator = RAGOrchestrator.from_pipeline(
            pipeline=self.rag_pipeline,
            security_kb=security_kb,
            top_k_code=top_k,
            top_k_security=self._retrieval_cfg.get("top_k_security", 3),
            semantic_weight=self._retrieval_cfg.get("semantic_weight", 0.7),
            keyword_weight=self._retrieval_cfg.get("keyword_weight", 0.3),
            cache_ttl=self._retrieval_cfg.get("cache_ttl", 300),
            max_context_tokens=self._retrieval_cfg.get("max_context_tokens", 3000),
        )
        orchestrator.context_builder = self.context_builder
        return orchestrator

    def _retrieve_code(self, query: str, top_k: int, metadata_filter: Optional[Dict] = None, category_filter: Optional[str] = None, file_type_filter: Optional[str] = None) -> List[Any]:
        if self._hybrid_retriever is None:
            return []
        results = self._hybrid_retriever.retrieve(query=query, top_k=max(top_k * 3, 10), metadata_filter=metadata_filter)
        filtered = []
        for result in results:
            chunk = result.chunk
            metadata = chunk.get("metadata", {})
            if category_filter and metadata.get("category") != category_filter:
                continue
            if file_type_filter and chunk.get("file_type") != file_type_filter:
                continue
            filtered.append(result)
            if len(filtered) >= top_k:
                break
        return filtered if filtered or not (category_filter or file_type_filter) else []

    @staticmethod
    def _is_iam_chunk(chunk: Dict[str, Any]) -> bool:
        metadata = chunk.get("metadata", {}) or {}
        file_type = chunk.get("file_type", "")
        resource_type = metadata.get("resource_type") or chunk.get("resource_type", "")
        text = (chunk.get("text", "") or "").lower()

        if file_type == "terraform":
            return resource_type == "azurerm_role_assignment"
        if file_type in {"yaml", "json"}:
            return "roleassignment" in text or "roledefinition" in text
        return False

    def _expand_iam_chunks(self, code_results: List[Any]) -> List[Dict[str, Any]]:
        retrieved_chunks = [result.chunk for result in code_results]
        file_paths = sorted(
            {
                chunk.get("file_path", "")
                for chunk in retrieved_chunks
                if chunk.get("file_path")
            }
        )

        vector_store = getattr(self.rag_pipeline, "vector_store", None)
        if vector_store is None or not hasattr(vector_store, "get_chunks_for_file"):
            return retrieved_chunks

        expanded_chunks: List[Dict[str, Any]] = []
        seen = set()
        for file_path in file_paths:
            for chunk in vector_store.get_chunks_for_file(file_path):
                if not self._is_iam_chunk(chunk):
                    continue
                chunk_id = chunk.get("chunk_id") or (
                    chunk.get("file_path", ""),
                    chunk.get("chunk_index", 0),
                    chunk.get("metadata", {}).get("resource_type", ""),
                )
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                expanded_chunks.append(chunk)

        return expanded_chunks or retrieved_chunks

    def _normalize_result(self, result: dict, code_results: Optional[List[Any]] = None, fallback_analysis: str = "") -> dict:
        findings = result.get("findings") or result.get("issues") or []
        evidence = result.get("evidence", {}) or {}
        code_chunks = list(evidence.get("code_chunks", []))
        if code_results:
            code_chunks = [res.to_dict() if hasattr(res, "to_dict") else res for res in code_results]

        sources = set()
        for item in code_chunks:
            payload = item.get("chunk", item)
            file_path = payload.get("file_path") or item.get("file_path")
            if file_path:
                sources.add(file_path)
        for finding in findings:
            file_path = finding.get("file_path")
            if file_path:
                sources.add(file_path)

        analysis = result.get("analysis") or fallback_analysis
        if not analysis:
            summary = result.get("summary", "")
            analysis = summary if isinstance(summary, str) else json.dumps(summary)

        normalized = {
            "chunks_retrieved": len(code_chunks),
            "analysis": analysis,
            "findings": findings,
            "issues": findings,
            "sources": sorted(sources),
            "context": result.get("context", ""),
            "results": [item.get("chunk", item) for item in code_chunks],
            "summary": result.get("summary", ""),
            "evidence": evidence if evidence else {"code_chunks": code_chunks, "security_references": []},
            "metadata": result.get("metadata", {}),
        }
        return normalized

    def _handle_iam(self, query: str, top_k: int, stream: bool = False) -> dict:
        if self.iam_analyzer is None:
            return self._handle_general(query, top_k, stream=stream, intent_hint="IAM over-permission and managed identity")
        code_results = self._retrieve_code(query, top_k, metadata_filter={"cloud_provider": "azure"}, category_filter="identity")
        iam_chunks = self._expand_iam_chunks(code_results)
        result = self.iam_analyzer.analyze_chunks(iam_chunks, query=query)
        return self._normalize_result(result, code_results=code_results)

    def _handle_workload_identity(self, query: str, top_k: int, stream: bool = False) -> dict:
        if self.workload_identity_analyzer is None:
            return self._handle_general(query, top_k, stream=stream, intent_hint="workload identity federation, OIDC trust, and subject claims")
        code_results = self._retrieve_code(query, top_k, category_filter="identity")
        result = self.workload_identity_analyzer.analyze_chunks([r.chunk for r in code_results], query=query)
        return self._normalize_result(result, code_results=code_results)

    def _handle_secrets(self, query: str, top_k: int, stream: bool = False) -> dict:
        if self.secrets_analyzer is None:
            return self._handle_general(query, top_k, stream=stream, intent_hint="hardcoded secrets, credentials, and sensitive values")
        code_results = self._retrieve_code(query, top_k, category_filter="secrets")
        result = self.secrets_analyzer.analyze_chunks([r.chunk for r in code_results], query=query)
        return self._normalize_result(result, code_results=code_results)

    def _handle_prompt_injection(self, query: str, top_k: int, stream: bool = False) -> dict:
        if self.prompt_injection_analyzer is None:
            return self._handle_general(query, top_k, stream=stream, intent_hint="prompt injection, workflow abuse, and CI/CD script risks")
        code_results = self._retrieve_code(query, top_k, file_type_filter="yaml")
        result = self.prompt_injection_analyzer.analyze_chunks([r.chunk for r in code_results], query=query)
        return self._normalize_result(result, code_results=code_results)

    def _handle_network(self, query: str, top_k: int, stream: bool = False) -> dict:
        return self._handle_general(query, top_k, stream=stream, metadata_filter=None, intent_hint="network exposure, firewall, and public access")

    def _handle_compliance(self, query: str, top_k: int, stream: bool = False) -> dict:
        return self._handle_general(query, top_k, stream=stream, metadata_filter=None, intent_hint="compliance, benchmark, and policy violations")

    def _handle_general(self, query: str, top_k: int, stream: bool = False, metadata_filter: Optional[Dict] = None, intent_hint: str = "general security") -> dict:
        if self._hybrid_retriever is None:
            if hasattr(self.rag_pipeline, "analyze"):
                result = self.rag_pipeline.analyze(query, top_k=top_k, stream=stream)
                return self._normalize_result(result)
            return self._normalize_result({"analysis": "", "results": [], "context": ""})
        orchestrator = self._build_orchestrator(top_k)
        result = orchestrator.analyze(
            query=query,
            metadata_filter=metadata_filter,
            stream=stream,
            top_k_code=top_k,
            top_k_security=self._retrieval_cfg.get("top_k_security", 3),
            intent_hint=intent_hint,
        )
        return self._normalize_result(result)