"""Normalized repository of security rules for rule-aware retrieval."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from security_kb import _BUILT_IN_RULES


_CATEGORY_BY_RESOURCE = {
    "azurerm_storage_account": "storage",
    "aws_s3_bucket": "storage",
    "google_storage_bucket": "storage",
    "aws_security_group": "networking",
    "azurerm_network_security_group": "networking",
    "google_compute_firewall": "networking",
    "kubernetes_pod": "compute",
    "google_compute_instance": "compute",
    "aws_instance": "compute",
    "azurerm_linux_virtual_machine": "compute",
    "azurerm_user_assigned_identity": "identity",
    "service_account": "identity",
    "storage": "storage",
    "networking": "networking",
    "compute": "compute",
    "identity": "identity",
    "secrets": "secrets",
    "general": "general",
}


class RuleRepository:
    """Load and normalize built-in and custom security rules."""

    def __init__(self, extra_rules: Optional[Iterable[Dict]] = None) -> None:
        self._rules = [self._normalize_rule(rule) for rule in list(_BUILT_IN_RULES) + list(extra_rules or [])]
        self._by_id = {rule["rule_id"]: rule for rule in self._rules}

    def list_rules(self) -> List[Dict]:
        return list(self._rules)

    def get_rule(self, rule_id: str) -> Optional[Dict]:
        return self._by_id.get(rule_id)

    def _normalize_rule(self, rule: Dict) -> Dict:
        probe = " ".join(
            str(rule.get(key, ""))
            for key in ("title", "description", "indicators", "references", "category")
        ).lower()

        resource_type = rule.get("resource_type") or self._infer_resource_type(probe)
        cloud_provider = rule.get("cloud_provider") or self._infer_cloud_provider(rule, probe)
        category = rule.get("rule_category") or rule.get("category")
        normalized_category = self._normalize_category(category, resource_type, probe)
        description = rule.get("description", "")

        return {
            "rule_id": rule.get("rule_id") or rule.get("id", ""),
            "title": rule.get("title", ""),
            "description": description,
            "resource_type": resource_type,
            "cloud_provider": cloud_provider,
            "category": normalized_category,
            "severity": str(rule.get("severity", "INFO")).upper(),
            "indicators": rule.get("indicators", ""),
            "remediation": rule.get("remediation", ""),
            "references": rule.get("references", ""),
            "source_category": rule.get("category", "general"),
        }

    def _infer_resource_type(self, probe: str) -> str:
        if "managed identity" in probe:
            return "azurerm_user_assigned_identity"
        if "storage account" in probe or "blob" in probe:
            return "azurerm_storage_account" if "azure" in probe else "storage"
        if "s3 bucket" in probe:
            return "aws_s3_bucket"
        if "security group" in probe:
            return "aws_security_group"
        if "network security group" in probe or "nsg" in probe:
            return "azurerm_network_security_group"
        if "gcp storage bucket" in probe or "uniform bucket-level access" in probe:
            return "google_storage_bucket"
        if "compute engine" in probe or "vm instance" in probe:
            return "google_compute_instance"
        if "kubernetes" in probe or "container" in probe or "pod" in probe:
            return "kubernetes_pod"
        if "publicly accessible cloud storage" in probe or "storage resources" in probe:
            return "storage"
        if "network segmentation" in probe or "http endpoint" in probe:
            return "networking"
        if "password" in probe or "secret" in probe or "token" in probe or "key vault" in probe:
            return "secrets"
        if "identity" in probe or "iam" in probe or "oidc" in probe or "rbac" in probe:
            return "identity"
        if "compute" in probe or "cluster" in probe or "lambda" in probe:
            return "compute"
        return "general"

    def _infer_cloud_provider(self, rule: Dict, probe: str) -> str:
        category = str(rule.get("category", "")).lower()
        if "azure" in category or "azure" in probe:
            return "azure"
        if "aws" in category or "aws" in probe or "s3" in probe:
            return "aws"
        if "gcp" in category or "google" in probe:
            return "gcp"
        if "k8s" in category or "kubernetes" in probe:
            return "kubernetes"
        return "generic"

    def _normalize_category(self, category: str, resource_type: str, probe: str) -> str:
        lowered = str(category or "").lower()
        if lowered in {"identity", "networking", "secrets", "compute", "storage"}:
            return lowered
        if resource_type in _CATEGORY_BY_RESOURCE:
            return _CATEGORY_BY_RESOURCE[resource_type]
        if "storage" in probe or "bucket" in probe or "blob" in probe:
            return "storage"
        if "secret" in probe or "password" in probe or "token" in probe:
            return "secrets"
        if "network" in probe or "firewall" in probe or "security group" in probe:
            return "networking"
        if "identity" in probe or "iam" in probe or "role" in probe:
            return "identity"
        if "compute" in probe or "vm" in probe or "pod" in probe or "container" in probe:
            return "compute"
        return "general"