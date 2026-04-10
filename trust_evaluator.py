"""TrustEvaluator — deterministic checks for workload identity trust boundaries."""

from __future__ import annotations

from typing import List, Tuple
from urllib.parse import urlparse

from workload_identity_parser import WorkloadIdentityConfig

TRUSTED_ISSUER_HOST_SUFFIXES = (
    "azure.com",
    "azmk8s.io",
    "kubernetes.default.svc",
)


def _is_trusted_host(host: str) -> bool:
    if not host:
        return False
    return any(host == suffix or host.endswith("." + suffix) for suffix in TRUSTED_ISSUER_HOST_SUFFIXES)


class TrustEvaluator:
    def evaluate(self, config: WorkloadIdentityConfig) -> List[Tuple[str, str, str, str]]:
        findings: List[Tuple[str, str, str, str]] = []
        if config.federation_type == "kubernetes_service_account" and not config.explicit_issuer:
            return findings

        issuer = (config.issuer or "").strip()
        parsed = urlparse(issuer) if issuer else None
        host = (parsed.hostname or "").lower() if parsed else ""

        broad_subject = "*" in (config.subject or "") or config.subject.endswith(":")
        trusted_host = _is_trusted_host(host)

        if issuer and host and not trusted_host and (broad_subject or not config.tenant_id):
            findings.append(
                (
                    "WID-AZ-004",
                    "HIGH",
                    "External or cross-tenant trust without restriction",
                    (
                        "Restrict federation to a trusted issuer and pin trust boundaries with "
                        "explicit tenant and subject constraints. Avoid wildcard trust for "
                        "external identity providers."
                    ),
                )
            )

        return findings