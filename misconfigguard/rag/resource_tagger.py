"""Resource tagging helpers for rule-aware retrieval."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List

from vector_store_manager import detect_cloud_provider


_TERRAFORM_RESOURCE_RE = re.compile(r'resource\s+"([^"]+)"\s+"([^"]+)"')
_TERRAFORM_DATA_RE = re.compile(r'data\s+"([^"]+)"\s+"([^"]+)"')
_YAML_KIND_RE = re.compile(r"^kind\s*:\s*([A-Za-z0-9_\-]+)", re.MULTILINE)

_CATEGORY_PATTERNS = [
    ("storage", ("storage", "bucket", "blob", "disk", "volume", "efs", "s3")),
    ("identity", ("identity", "iam", "role", "rbac", "oidc", "service_account", "principal")),
    ("networking", ("network", "subnet", "security_group", "firewall", "nsg", "vpc", "load_balancer", "gateway")),
    ("secrets", ("secret", "key_vault", "kms", "secret_manager", "credential", "password", "token")),
    ("compute", ("instance", "vm", "compute", "lambda", "function", "pod", "container", "cluster", "deployment")),
]

_PROVIDER_ALIASES = {
    "aws": "aws",
    "azure": "azure",
    "gcp": "gcp",
    "google": "gcp",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "unknown": "unknown",
    "": "unknown",
}


class ResourceTagger:
    """Infer resource_type, cloud_provider, and category for code chunks."""

    def tag_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        tagged = copy.deepcopy(chunk)
        metadata = dict(tagged.get("metadata", {}))

        resource_type = (
            tagged.get("resource_type")
            or metadata.get("resource_type")
            or self._infer_resource_type(tagged)
        )
        cloud_provider = self._normalize_provider(
            tagged.get("cloud_provider")
            or metadata.get("cloud_provider")
            or detect_cloud_provider(
                resource_type=resource_type,
                content=tagged.get("text", ""),
                file_path=tagged.get("file_path", ""),
            )
        )
        category = metadata.get("category") or self._infer_category(resource_type, tagged.get("text", ""))

        tagged["resource_type"] = resource_type
        tagged["cloud_provider"] = cloud_provider
        metadata["resource_type"] = resource_type
        metadata["cloud_provider"] = cloud_provider
        if category:
            metadata["category"] = category
        tagged["metadata"] = metadata
        return tagged

    def tag_chunks(self, chunks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.tag_chunk(chunk) for chunk in chunks]

    def extract_resource_matches(self, results: Iterable[Any]) -> List[Dict[str, str]]:
        matches: List[Dict[str, str]] = []
        seen = set()
        for result in results:
            chunk = getattr(result, "chunk", result)
            tagged = self.tag_chunk(chunk)
            metadata = tagged.get("metadata", {})
            resource_type = metadata.get("resource_type", "")
            cloud_provider = metadata.get("cloud_provider", "unknown")
            category = metadata.get("category", "")
            key = (resource_type, cloud_provider, category)
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                {
                    "resource_type": resource_type,
                    "cloud_provider": cloud_provider,
                    "category": category,
                }
            )
        return matches

    def _infer_resource_type(self, chunk: Dict[str, Any]) -> str:
        text = chunk.get("text", "")
        file_type = str(chunk.get("file_type", "")).lower()
        if file_type == "terraform":
            match = _TERRAFORM_RESOURCE_RE.search(text) or _TERRAFORM_DATA_RE.search(text)
            if match:
                return match.group(1)
        if file_type in {"yaml", "yml", "json"}:
            match = _YAML_KIND_RE.search(text)
            if match:
                return match.group(1)
        return ""

    def _infer_category(self, resource_type: str, text: str) -> str:
        probe = f"{resource_type} {text}".lower()
        for category, keywords in _CATEGORY_PATTERNS:
            if any(keyword in probe for keyword in keywords):
                return category
        return "general"

    def _normalize_provider(self, provider: str) -> str:
        return _PROVIDER_ALIASES.get((provider or "").lower(), (provider or "unknown").lower())