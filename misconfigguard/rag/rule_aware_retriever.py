"""Rule-aware retrieval that narrows security guidance to matched resources."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from resource_tagger import ResourceTagger
from security_kb import SecurityRuleResult
from rule_filter import RuleFilter
from rule_repository import RuleRepository


class RuleAwareRetriever:
    """Retrieve code chunks first, then retrieve only matching security rules."""

    def __init__(
        self,
        code_retriever: Any,
        security_kb: Any,
        embedder: Any,
        resource_tagger: Optional[ResourceTagger] = None,
        rule_repository: Optional[RuleRepository] = None,
        rule_filter: Optional[RuleFilter] = None,
    ) -> None:
        self.code_retriever = code_retriever
        self.security_kb = security_kb
        self.embedder = embedder
        self.resource_tagger = resource_tagger or ResourceTagger()
        self.rule_repository = (
            rule_repository
            or getattr(security_kb, "_repository", None)
            or RuleRepository()
        )
        self.rule_filter = rule_filter or RuleFilter()

    def retrieve(
        self,
        query: str,
        top_k_code: int = 5,
        top_k_rules: int = 3,
        metadata_filter: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        code_results = self.code_retriever.retrieve(
            query=query,
            top_k=top_k_code,
            metadata_filter=metadata_filter,
        )
        matched_resources = self.resource_tagger.extract_resource_matches(code_results)
        if not matched_resources:
            return {
                "code_results": code_results,
                "security_results": self._fallback_rule_search(query, top_k_rules),
                "matched_resources": [],
            }

        normalized_rules = self.rule_repository.list_rules()
        merged: Dict[str, Any] = {}

        for resource in matched_resources:
            applicable_rules = self.rule_filter.filter_rules(
                rules=normalized_rules,
                resource_types=[resource.get("resource_type", "")],
                cloud_provider=resource.get("cloud_provider", "generic"),
                category=resource.get("category", "general"),
            )
            for rule in applicable_rules:
                synthesized = self._result_from_rule(rule, resource, len(merged) + 1)
                current = merged.get(synthesized.rule_id)
                if current is None or synthesized.score > current.score:
                    merged[synthesized.rule_id] = synthesized

        if not merged:
            security_results = self._fallback_rule_search(query, top_k_rules)
        else:
            security_results = sorted(
                merged.values(),
                key=lambda result: (-result.score, result.rank, result.rule_id),
            )[:top_k_rules]

        return {
            "code_results": code_results,
            "security_results": security_results,
            "matched_resources": matched_resources,
        }

    def _embed_query(self, query: str):
        if hasattr(self.code_retriever, "_embed_query"):
            return self.code_retriever._embed_query(query)
        return self.embedder.embed([query])[0]

    def _fallback_rule_search(self, query: str, top_k_rules: int) -> List[Any]:
        return self.security_kb.search(self._embed_query(query), top_k=top_k_rules)

    def _result_from_rule(self, rule: Dict[str, Any], resource: Dict[str, str], rank: int) -> SecurityRuleResult:
        resource_type = rule.get("resource_type", "general")
        matched_resource_type = resource.get("resource_type", "")
        score = 0.95 if resource_type == matched_resource_type else 0.85
        if rule.get("rule_id", "").startswith("custom-"):
            score += 0.2
        text = rule.get("text") or self.security_kb._rule_to_text(rule)
        return SecurityRuleResult(
            rule_id=rule.get("rule_id", ""),
            title=rule.get("title", ""),
            severity=rule.get("severity", "INFO"),
            category=rule.get("category", "general"),
            description=rule.get("description", ""),
            resource_type=resource_type,
            cloud_provider=rule.get("cloud_provider", "generic"),
            text=text,
            score=score,
            rank=rank,
        )