"""Filtering logic for matching security rules to tagged resources."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


_SEVERITY_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}


class RuleFilter:
    """Select rules relevant to a set of tagged resources."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[Tuple[str, ...], str, str, Tuple[str, ...]], List[Dict]] = {}

    def filter_rules(
        self,
        rules: Iterable[Dict],
        resource_types: Sequence[str],
        cloud_provider: str,
        category: str = "",
    ) -> List[Dict]:
        rule_list = list(rules)
        normalized_resource_types = tuple(sorted({resource_type for resource_type in resource_types if resource_type}))
        rule_signature = tuple(sorted(str(rule.get("rule_id", "")) for rule in rule_list))
        key = (
            normalized_resource_types,
            cloud_provider or "generic",
            category or "",
            rule_signature,
        )
        if key in self._cache:
            return list(self._cache[key])

        filtered = [
            rule
            for rule in rule_list
            if self._provider_matches(rule.get("cloud_provider", "generic"), cloud_provider)
            and self._resource_matches(rule.get("resource_type", "general"), normalized_resource_types, category)
        ]
        filtered.sort(key=lambda rule: (_SEVERITY_RANK.get(rule.get("severity", "INFO"), 9), rule.get("rule_id", "")))
        self._cache[key] = list(filtered)
        return filtered

    def _provider_matches(self, rule_provider: str, cloud_provider: str) -> bool:
        normalized_rule_provider = (rule_provider or "generic").lower()
        normalized_cloud_provider = (cloud_provider or "generic").lower()
        return normalized_rule_provider in {normalized_cloud_provider, "generic", "all", "unknown"}

    def _resource_matches(self, rule_resource_type: str, resource_types: Sequence[str], category: str) -> bool:
        normalized_rule_type = (rule_resource_type or "general").lower()
        normalized_resources = {resource_type.lower() for resource_type in resource_types}
        normalized_category = (category or "general").lower()
        if normalized_rule_type in normalized_resources:
            return True
        if normalized_category and normalized_rule_type == normalized_category:
            return True
        return not normalized_resources and normalized_rule_type == "general"