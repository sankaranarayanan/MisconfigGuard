"""
WorkloadIdentityParser — Extract workload identity federation configuration
from Terraform, Kubernetes YAML, and ARM-style JSON/YAML documents.

Produces normalised WorkloadIdentityConfig objects consumed by
FederationAnalyzer and WorkloadIdentitySecurityAnalyzer.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from iam_parser import _extract_tf_resource_blocks, _get_tf_attr

logger = logging.getLogger(__name__)


@dataclass
class WorkloadIdentityConfig:
    identity: str
    identity_type: str
    federation_type: str
    issuer: str
    subject: str
    audiences: List[str] = field(default_factory=list)
    tenant_id: str = ""
    provider: str = "azure"
    namespace: str = ""
    service_account: str = ""
    explicit_issuer: bool = True
    explicit_subject: bool = True
    explicit_audiences: bool = True
    file_path: str = ""
    block_name: str = ""
    source_text: str = ""

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "identity_type": self.identity_type,
            "federation_type": self.federation_type,
            "issuer": self.issuer,
            "subject": self.subject,
            "audiences": list(self.audiences),
            "tenant_id": self.tenant_id,
            "provider": self.provider,
            "namespace": self.namespace,
            "service_account": self.service_account,
            "explicit_issuer": self.explicit_issuer,
            "explicit_subject": self.explicit_subject,
            "explicit_audiences": self.explicit_audiences,
            "file_path": self.file_path,
            "block_name": self.block_name,
        }


def _parse_subject(subject: str) -> tuple[str, str]:
    if not subject:
        return "", ""
    parts = subject.split(":")
    if len(parts) >= 4 and parts[0] == "system" and parts[1] == "serviceaccount":
        return parts[2], parts[3]
    return "", ""


def _infer_identity_from_ref(value: str, block_name: str) -> str:
    if not value:
        return block_name
    match = re.search(r"\w+\.(\w+)\.(?:id|object_id|application_id)\b", value)
    if match:
        return match.group(1)
    return block_name


def _get_tf_list_attr(block: str, attr: str) -> List[str]:
    match = re.search(
        r"^\s*" + re.escape(attr) + r"\s*=\s*\[(.*?)\]",
        block,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        scalar = _get_tf_attr(block, attr)
        return [scalar] if scalar else []
    values = []
    for item in match.group(1).split(","):
        cleaned = item.strip().strip('"\'')
        if cleaned:
            values.append(cleaned)
    return values


def _split_annotation_audience(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class WorkloadIdentityParser:
    def parse_terraform(self, content: str, file_path: str = "") -> List[WorkloadIdentityConfig]:
        results: List[WorkloadIdentityConfig] = []
        resource_types = (
            "azurerm_federated_identity_credential",
            "azuread_application_federated_identity_credential",
        )

        for resource_type in resource_types:
            for block_name, block_text, _ in _extract_tf_resource_blocks(content, resource_type):
                issuer = _get_tf_attr(block_text, "issuer") or ""
                subject = _get_tf_attr(block_text, "subject") or ""
                audiences = _get_tf_list_attr(block_text, "audience") or _get_tf_list_attr(block_text, "audiences")
                identity_ref = (
                    _get_tf_attr(block_text, "parent_id")
                    or _get_tf_attr(block_text, "application_object_id")
                    or ""
                )
                namespace, service_account = _parse_subject(subject)
                full_text = (
                    f'resource "{resource_type}" "{block_name}" {{\n'
                    f"{block_text}\n}}"
                )
                results.append(
                    WorkloadIdentityConfig(
                        identity=_infer_identity_from_ref(identity_ref, block_name),
                        identity_type="workload_identity",
                        federation_type="azure_federated_credential",
                        issuer=issuer,
                        subject=subject,
                        audiences=audiences,
                        tenant_id=_get_tf_attr(block_text, "tenant_id") or "",
                        provider="azure",
                        namespace=namespace,
                        service_account=service_account,
                        explicit_issuer=True,
                        explicit_subject=True,
                        explicit_audiences=True,
                        file_path=file_path,
                        block_name=block_name,
                        source_text=full_text[:800],
                    )
                )

        return results

    def parse_yaml(self, content: str, file_path: str = "") -> List[WorkloadIdentityConfig]:
        try:
            documents = list(yaml.safe_load_all(content))
        except Exception as exc:
            logger.debug("Workload identity YAML parse failed (%s): %s", file_path, exc)
            return []
        results: List[WorkloadIdentityConfig] = []
        for document in documents:
            results.extend(self._parse_data(document, file_path))
        return results

    def parse_json(self, content: str, file_path: str = "") -> List[WorkloadIdentityConfig]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.debug("Workload identity JSON parse failed (%s): %s", file_path, exc)
            return []
        return self._parse_data(data, file_path)

    def parse_chunks(self, chunks: List[dict]) -> List[WorkloadIdentityConfig]:
        results: List[WorkloadIdentityConfig] = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            file_type = chunk.get("file_type", "")
            file_path = chunk.get("file_path", "")
            resource_type = meta.get("resource_type", "")
            text = chunk.get("text", "")

            if file_type == "terraform" and resource_type in {
                "azurerm_federated_identity_credential",
                "azuread_application_federated_identity_credential",
            }:
                results.extend(self.parse_terraform(text, file_path))
            elif file_type in {"yaml", "json"} and text:
                lowered = text.lower()
                if (
                    "federatedidentitycredential" in lowered
                    or "azure.workload.identity/" in lowered
                    or "kind: serviceaccount" in lowered
                ):
                    parser = self.parse_yaml if file_type == "yaml" else self.parse_json
                    results.extend(parser(text, file_path))

        deduped: List[WorkloadIdentityConfig] = []
        seen = set()
        for config in results:
            key = (
                config.file_path,
                config.block_name,
                config.identity,
                config.subject,
                tuple(config.audiences),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(config)
        return deduped

    def parse_file(self, file_path: str) -> List[WorkloadIdentityConfig]:
        path = Path(file_path)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Could not read %s: %s", file_path, exc)
            return []

        if path.suffix.lower() == ".tf":
            return self.parse_terraform(content, file_path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return self.parse_yaml(content, file_path)
        if path.suffix.lower() == ".json":
            return self.parse_json(content, file_path)
        return []

    def _parse_data(self, data: object, file_path: str) -> List[WorkloadIdentityConfig]:
        results: List[WorkloadIdentityConfig] = []
        if isinstance(data, dict):
            self._visit_node(data, file_path, results)
        elif isinstance(data, list):
            for item in data:
                results.extend(self._parse_data(item, file_path))
        return results

    def _visit_node(self, node: dict, file_path: str, results: List[WorkloadIdentityConfig]) -> None:
        kind = str(node.get("kind", "")).lower()
        resource_type = str(node.get("type") or node.get("Type") or "").lower()

        if kind == "serviceaccount":
            config = self._service_account_to_config(node, file_path)
            if config:
                results.append(config)

        if "federatedidentitycredentials" in resource_type:
            config = self._arm_federation_to_config(node, file_path)
            if config:
                results.append(config)

        for value in node.values():
            if isinstance(value, dict):
                self._visit_node(value, file_path, results)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._visit_node(item, file_path, results)

    def _service_account_to_config(
        self,
        node: dict,
        file_path: str,
    ) -> Optional[WorkloadIdentityConfig]:
        metadata = node.get("metadata") or {}
        annotations = metadata.get("annotations") or {}
        if not any(key.startswith("azure.workload.identity/") for key in annotations):
            return None

        name = str(metadata.get("name", ""))
        namespace = str(metadata.get("namespace", "default"))
        explicit_issuer = "azure.workload.identity/issuer" in annotations
        explicit_subject = "azure.workload.identity/subject" in annotations
        explicit_audiences = "azure.workload.identity/audience" in annotations
        issuer = str(annotations.get("azure.workload.identity/issuer", ""))
        subject = str(annotations.get("azure.workload.identity/subject", ""))
        audience_value = str(annotations.get("azure.workload.identity/audience", ""))
        return WorkloadIdentityConfig(
            identity=name,
            identity_type="workload_identity",
            federation_type="kubernetes_service_account",
            issuer=issuer,
            subject=subject,
            audiences=_split_annotation_audience(audience_value),
            tenant_id=str(annotations.get("azure.workload.identity/tenant-id", "")),
            provider="azure",
            namespace=namespace,
            service_account=name,
            explicit_issuer=explicit_issuer,
            explicit_subject=explicit_subject,
            explicit_audiences=explicit_audiences,
            file_path=file_path,
            block_name=name,
            source_text=yaml.safe_dump(node, sort_keys=False)[:800],
        )

    def _arm_federation_to_config(
        self,
        node: dict,
        file_path: str,
    ) -> Optional[WorkloadIdentityConfig]:
        props = node.get("properties") or {}
        issuer = str(props.get("issuer") or props.get("Issuer") or "")
        subject = str(props.get("subject") or props.get("Subject") or "")
        raw_audiences = props.get("audiences") or props.get("Audiences") or []
        audiences = [str(item) for item in raw_audiences if str(item)]
        namespace, service_account = _parse_subject(subject)
        name = str(node.get("name", ""))
        return WorkloadIdentityConfig(
            identity=name,
            identity_type="workload_identity",
            federation_type="azure_federated_credential",
            issuer=issuer,
            subject=subject,
            audiences=audiences,
            tenant_id=str(props.get("tenantId") or props.get("tenantID") or ""),
            provider="azure",
            namespace=namespace,
            service_account=service_account,
            explicit_issuer=True,
            explicit_subject=True,
            explicit_audiences=True,
            file_path=file_path,
            block_name=name,
            source_text=json.dumps(node, indent=2)[:800],
        )