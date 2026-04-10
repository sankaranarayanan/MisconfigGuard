from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from routing_cache import RoutingCache

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    GENERAL_SECURITY = "general_security"
    IAM = "iam"
    WORKLOAD_IDENTITY = "workload_identity"
    SECRETS = "secrets"
    PROMPT_INJECTION = "prompt_injection"
    NETWORK = "network"
    COMPLIANCE = "compliance"


class QueryRouter:
    def __init__(
        self,
        llm_client=None,
        use_llm_routing: bool = True,
        routing_model: Optional[str] = None,
        routing_max_tokens: int = 20,
        cache_ttl: int = 300,
    ):
        self.llm_client = llm_client
        self.use_llm_routing = use_llm_routing
        self.routing_model = routing_model
        self.routing_max_tokens = routing_max_tokens
        self.cache = RoutingCache(ttl=cache_ttl) if cache_ttl > 0 else None

    def classify(self, query: str) -> QueryIntent:
        cached = self.cache.get(query) if self.cache else None
        if cached is not None:
            return cached

        intent = None
        if self.use_llm_routing:
            intent = self._llm_classify(query)
        if intent is None:
            intent = self._keyword_classify(query)

        if self.cache is not None:
            self.cache.set(query, intent)
        return intent

    def _llm_classify(self, query: str) -> Optional[QueryIntent]:
        if self.llm_client is None or not hasattr(self.llm_client, "generate"):
            return None
        if hasattr(self.llm_client, "is_available") and not self.llm_client.is_available():
            return None

        prompt = (
            "You are a security query classifier. Given a user question about\n"
            "infrastructure-as-code security, respond with EXACTLY ONE label from\n"
            "this list and nothing else:\n"
            "general_security | iam | workload_identity | secrets |\n"
            "prompt_injection | network | compliance\n\n"
            f"USER: {query}"
        )

        original_model = getattr(self.llm_client, "model", None)
        original_max_tokens = getattr(self.llm_client, "max_tokens", None)
        try:
            if self.routing_model:
                self.llm_client.model = self.routing_model
            if original_max_tokens is not None:
                self.llm_client.max_tokens = self.routing_max_tokens
            raw = self.llm_client.generate(prompt)
        except Exception as exc:
            logger.info("LLM routing failed, falling back to keywords: %s", exc)
            return None
        finally:
            if original_model is not None:
                self.llm_client.model = original_model
            if original_max_tokens is not None:
                self.llm_client.max_tokens = original_max_tokens

        token = (raw or "").strip().strip("\"'").lower()
        for intent in QueryIntent:
            if token == intent.value:
                return intent
        return QueryIntent.GENERAL_SECURITY

    def _keyword_classify(self, query: str) -> QueryIntent:
        probe = (query or "").lower()
        mappings = [
            (QueryIntent.IAM, ["managed identity", "managed identities", "role assignment", "contributor", "owner role", "rbac", "service principal", "iam", "permission"]),
            (QueryIntent.WORKLOAD_IDENTITY, ["oidc", "workload identity", "federation", "issuer", "audience", "subject claim", "trust policy"]),
            (QueryIntent.SECRETS, ["secret", "password", "api key", "token", "credential", "hardcoded", "entropy", "private key", "connection string"]),
            (QueryIntent.PROMPT_INJECTION, ["prompt injection", "pipeline", "github actions", "azure devops", "ci/cd", "workflow yaml", "run:", "script:"]),
            (QueryIntent.NETWORK, ["security group", "open port", "0.0.0.0", "ingress", "egress", "firewall", "public access", "acl"]),
            (QueryIntent.COMPLIANCE, ["cis", "compliance", "benchmark", "policy", "standard", "regulation", "pci", "hipaa", "nist"]),
        ]
        for intent, keywords in mappings:
            if any(keyword in probe for keyword in keywords):
                return intent
        return QueryIntent.GENERAL_SECURITY