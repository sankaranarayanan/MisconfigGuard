"""
FederationAnalyzer — Rule-based workload identity federation validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse

from trust_evaluator import TrustEvaluator
from workload_identity_parser import WorkloadIdentityConfig

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"


@dataclass
class WorkloadIdentityFinding:
    rule_id: str
    identity: str
    type: str
    severity: str
    issue: str
    explanation: str
    fix: str
    issuer: str = ""
    subject: str = ""
    audiences: List[str] | None = None
    file_path: str = ""
    block_name: str = ""

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "identity": self.identity,
            "type": self.type,
            "severity": self.severity,
            "issue": self.issue,
            "explanation": self.explanation,
            "fix": self.fix,
            "issuer": self.issuer,
            "subject": self.subject,
            "audiences": list(self.audiences or []),
            "file_path": self.file_path,
            "block_name": self.block_name,
        }


def _valid_https_url(value: str) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _broad_subject(subject: str) -> bool:
    if not subject:
        return False
    if "*" in subject:
        return True
    if re.fullmatch(r"system:serviceaccount:[^:]+", subject):
        return True
    return False


class FederationAnalyzer:
    def __init__(self, trust_evaluator: TrustEvaluator | None = None) -> None:
        self.trust_evaluator = trust_evaluator or TrustEvaluator()

    def analyze(self, configs: List[WorkloadIdentityConfig]) -> List[WorkloadIdentityFinding]:
        findings: List[WorkloadIdentityFinding] = []

        for config in configs:
            enforce_strict_claims = not (
                config.federation_type == "kubernetes_service_account"
                and not any((config.explicit_issuer, config.explicit_subject, config.explicit_audiences))
            )

            if enforce_strict_claims and config.explicit_issuer and not _valid_https_url(config.issuer):
                findings.append(
                    self._finding(
                        config,
                        "WID-AZ-001",
                        HIGH,
                        "Invalid or missing federation issuer",
                        "The workload identity federation issuer is missing or malformed. Azure AD cannot safely validate tokens from an untrusted or invalid issuer.",
                        "Define an explicit HTTPS issuer URL for the trusted OIDC provider and verify that it matches the cluster or external IdP exactly.",
                    )
                )

            if enforce_strict_claims and config.explicit_audiences and (
                not config.audiences or any(aud.strip() in {"", "*"} for aud in config.audiences)
            ):
                findings.append(
                    self._finding(
                        config,
                        "WID-AZ-002",
                        HIGH,
                        "Missing or overly broad audience restriction",
                        "The federated credential does not restrict the token audience tightly enough. Tokens intended for other audiences may be accepted by mistake.",
                        "Define explicit audiences such as api://AzureADTokenExchange and remove wildcard or empty values.",
                    )
                )

            if config.explicit_subject and _broad_subject(config.subject):
                findings.append(
                    self._finding(
                        config,
                        "WID-AZ-003",
                        MEDIUM,
                        "Overly broad subject claim",
                        "The subject claim allows more identities than a single workload instance, which weakens federation boundaries and increases the chance of privilege escalation.",
                        "Restrict subject to the exact workload identity, for example system:serviceaccount:<namespace>:<service-account>.",
                    )
                )

            for rule_id, severity, issue, fix in self.trust_evaluator.evaluate(config):
                findings.append(
                    self._finding(
                        config,
                        rule_id,
                        severity,
                        issue,
                        "The trust relationship permits identities outside the intended security boundary or relies on an issuer that is not sufficiently restricted.",
                        fix,
                    )
                )

        order = {HIGH: 0, MEDIUM: 1, LOW: 2}
        findings.sort(key=lambda item: order.get(item.severity, 9))
        return findings

    def _finding(
        self,
        config: WorkloadIdentityConfig,
        rule_id: str,
        severity: str,
        issue: str,
        explanation: str,
        fix: str,
    ) -> WorkloadIdentityFinding:
        return WorkloadIdentityFinding(
            rule_id=rule_id,
            identity=config.identity,
            type="workload_identity",
            severity=severity,
            issue=issue,
            explanation=explanation,
            fix=fix,
            issuer=config.issuer,
            subject=config.subject,
            audiences=config.audiences,
            file_path=config.file_path,
            block_name=config.block_name,
        )