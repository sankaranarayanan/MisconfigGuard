"""
PermissionAnalyzer — Rule-based over-permission detection for Azure managed identities.

Runs deterministic checks *before* (and independent of) any LLM reasoning,
ensuring that findings are always grounded in concrete security rules and are
reproducible regardless of whether a local LLM is available.

Rule set
--------
IAM-AZ-001  Owner / User Access Administrator at subscription scope     → HIGH
IAM-AZ-002  Contributor at subscription scope                           → HIGH
IAM-AZ-003  Contributor at resource group scope                         → MEDIUM
IAM-AZ-004  Owner at resource group or resource scope                   → HIGH
IAM-AZ-005  Identity assigned more than N roles (configurable)          → MEDIUM
IAM-AZ-006  Unknown scope with high-privilege role                      → MEDIUM
IAM-AZ-007  Overlapping / redundant role assignments                    → MEDIUM
IAM-AZ-008  Same role assigned at unnecessarily broad scope             → MEDIUM/LOW
IAM-AZ-009  Broad service-wide access role at broad scope               → MEDIUM/LOW

RecommendationEngine
--------------------
Maps each high-privilege role to a prioritised list of least-privilege
alternatives and suggests the next-narrower scope level.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from iam_parser import RoleAssignment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity constants
# ---------------------------------------------------------------------------

HIGH   = "HIGH"
MEDIUM = "MEDIUM"
LOW    = "LOW"

# ---------------------------------------------------------------------------
# Role categorisation
# ---------------------------------------------------------------------------

HIGH_PRIVILEGE_ROLES = frozenset({
    "Owner",
    "User Access Administrator",
    "Co-Administrator",
})

MEDIUM_PRIVILEGE_ROLES = frozenset({
    "Contributor",
    "Network Contributor",
    "Storage Account Contributor",
    "SQL Security Manager",
    "Security Admin",
    "Virtual Machine Administrator Login",
})

SERVICE_WIDE_ACCESS_ROLES = frozenset({
    "Storage Account Contributor",
    "Storage Blob Data Contributor",
    "Storage Queue Data Contributor",
    "Virtual Machine Contributor",
    "Network Contributor",
    "Key Vault Secrets Officer",
    "Monitoring Contributor",
    "SQL DB Contributor",
})

ROLE_PRIVILEGE_ORDER = {
    "Owner": 4,
    "User Access Administrator": 4,
    "Co-Administrator": 4,
    "Contributor": 3,
    "Storage Account Contributor": 3,
    "Virtual Machine Contributor": 3,
    "Network Contributor": 3,
    "Security Admin": 3,
    "Key Vault Secrets Officer": 3,
    "Storage Blob Data Contributor": 2,
    "Storage Queue Data Contributor": 2,
    "Reader": 1,
    "Storage Blob Data Reader": 1,
    "Key Vault Reader": 1,
}

_SCOPE_ORDER = {
    "subscription": 0,
    "resource_group": 1,
    "resource": 2,
    "unknown": 3,
}

# ---------------------------------------------------------------------------
# RecommendationEngine
# ---------------------------------------------------------------------------

# Ordered: most preferred alternative first
_ROLE_ALTERNATIVES: Dict[str, List[str]] = {
    "Owner": [
        "Contributor",
        "Reader",
        "User Access Administrator",  # only if IAM mgmt is truly required
    ],
    "User Access Administrator": [
        "Reader",
        "Contributor",
    ],
    "Co-Administrator": [
        "Contributor",
        "Reader",
    ],
    "Contributor": [
        "Reader",
        "Storage Blob Data Contributor",
        "Virtual Machine Contributor",
        "Key Vault Secrets User",
        "Network Contributor",
        "SQL DB Contributor",
    ],
    "Network Contributor": [
        "Reader",
        "Network Watcher Contributor",
    ],
    "Storage Account Contributor": [
        "Storage Blob Data Contributor",
        "Storage Blob Data Reader",
        "Storage Queue Data Contributor",
    ],
    "Virtual Machine Administrator Login": [
        "Virtual Machine User Login",
        "Virtual Machine Contributor",
    ],
    "Security Admin": [
        "Security Reader",
    ],
}

_SCOPE_REDUCTION: Dict[str, str] = {
    "subscription":  "resource_group",
    "resource_group": "resource",
    "resource":       "resource",   # already at narrowest
    "unknown":        "resource_group",
}


class RecommendationEngine:
    """
    Maps high-privilege roles and broad scopes to least-privilege alternatives.
    """

    def suggest_roles(self, role: str, context_hint: str = "") -> List[str]:
        """
        Return up to 4 recommended least-privilege alternatives for *role*.

        *context_hint* is optional text (e.g., identity name or resource type)
        used to surface the most contextually relevant alternatives first.
        """
        alternatives = list(_ROLE_ALTERNATIVES.get(role, ["Reader"]))
        if context_hint:
            hint = context_hint.lower().split()
            prioritised = [a for a in alternatives
                           if any(kw in a.lower() for kw in hint)]
            rest        = [a for a in alternatives if a not in prioritised]
            alternatives = prioritised + rest
        return alternatives[:4]

    def suggest_scope(self, current_scope_type: str) -> str:
        """Return the recommended narrower scope type."""
        return _SCOPE_REDUCTION.get(current_scope_type, "resource_group")

    def scope_guidance(self, current_scope: str, resource_hint: str = "") -> str:
        """Human-readable scope-reduction guidance string."""
        narrower = self.suggest_scope(current_scope)
        if narrower == current_scope:
            return "Scope is already at resource level — no further reduction possible."
        suffix = f" (e.g., the {resource_hint} resource group)" if resource_hint else ""
        return f"Reduce scope from {current_scope} to {narrower} level{suffix}."


# ---------------------------------------------------------------------------
# IAMFinding dataclass
# ---------------------------------------------------------------------------


@dataclass
class IAMFinding:
    """A single over-permission finding for an Azure role assignment."""

    rule_id        : str
    identity       : str
    role           : str
    scope          : str   # "subscription" | "resource_group" | "resource" | "unknown"
    scope_value    : str
    severity       : str   # "HIGH" | "MEDIUM" | "LOW"
    issue          : str   # Short title
    explanation    : str   # Detailed explanation
    fix            : str   # Concrete remediation
    recommendations: List[str] = field(default_factory=list)  # Suggested roles
    file_path      : str = ""
    block_name     : str = ""

    def to_dict(self) -> dict:
        return {
            "rule_id":         self.rule_id,
            "identity":        self.identity,
            "role":            self.role,
            "scope":           self.scope,
            "scope_value":     self.scope_value,
            "severity":        self.severity,
            "issue":           self.issue,
            "explanation":     self.explanation,
            "fix":             self.fix,
            "recommendations": self.recommendations,
            "file_path":       self.file_path,
            "block_name":      self.block_name,
        }


def _is_scope_broader(left: str, right: str) -> bool:
    return _SCOPE_ORDER.get(left, 99) < _SCOPE_ORDER.get(right, 99)


def _scope_contains(broad_scope_value: str, narrow_scope_value: str) -> bool:
    broad = (broad_scope_value or "").rstrip("/").lower()
    narrow = (narrow_scope_value or "").rstrip("/").lower()
    if not broad or not narrow:
        return False
    return narrow == broad or narrow.startswith(f"{broad}/")


def _role_supersedes(left: str, right: str) -> bool:
    if left == right:
        return True
    return ROLE_PRIVILEGE_ORDER.get(left, 0) > ROLE_PRIVILEGE_ORDER.get(right, 0)


# ---------------------------------------------------------------------------
# Internal rule helpers  (return None if rule does not apply)
# ---------------------------------------------------------------------------


def _r_owner_at_subscription(a: RoleAssignment) -> Optional[tuple]:
    """IAM-AZ-001: Owner / UAA at subscription scope → HIGH."""
    if a.role in HIGH_PRIVILEGE_ROLES and a.scope_type in ("subscription", "unknown"):
        return (
            "IAM-AZ-001",
            HIGH,
            f"'{a.role}' assigned at subscription scope",
            (
                f"Identity '{a.identity_name}' holds '{a.role}' at subscription scope, "
                f"granting unrestricted control over ALL resources in the subscription. "
                f"This is the highest-risk IAM configuration — a compromised identity "
                f"can escalate privileges, delete resources, or exfiltrate all data."
            ),
            (
                "Remove the subscription-level assignment immediately. "
                "Replace with a least-privilege, service-specific role scoped to the "
                "specific resource group or individual resource that requires access."
            ),
        )
    return None


def _r_contributor_at_subscription(a: RoleAssignment) -> Optional[tuple]:
    """IAM-AZ-002: Contributor at subscription scope → HIGH."""
    if a.role == "Contributor" and a.scope_type in ("subscription", "unknown"):
        return (
            "IAM-AZ-002",
            HIGH,
            "Contributor role assigned at subscription scope",
            (
                f"Identity '{a.identity_name}' has Contributor at subscription scope. "
                f"Contributor can create, read, update, and delete ANY resource "
                f"across the entire subscription (excluding IAM). The blast radius "
                f"of a compromised Contributor identity is extremely large."
            ),
            (
                "Scope the Contributor assignment to the specific resource group "
                "that requires management access. Better: replace with a "
                "service-specific role (e.g., Storage Blob Data Contributor for "
                "storage workloads, Virtual Machine Contributor for VM management)."
            ),
        )
    return None


def _r_contributor_at_rg(a: RoleAssignment) -> Optional[tuple]:
    """IAM-AZ-003: Contributor at resource group scope → MEDIUM."""
    if a.role == "Contributor" and a.scope_type == "resource_group":
        return (
            "IAM-AZ-003",
            MEDIUM,
            "Contributor role at resource group scope",
            (
                f"Identity '{a.identity_name}' has Contributor on a resource group. "
                f"This allows creating, modifying, and deleting every resource in the "
                f"group, which is broader than typically needed for single-purpose "
                f"managed identities."
            ),
            (
                "Replace Contributor with a service-specific role covering only "
                "the resource types this identity manages. "
                "Examples: Storage Blob Data Contributor, Virtual Machine Contributor, "
                "Key Vault Secrets User."
            ),
        )
    return None


def _r_owner_at_other_scope(a: RoleAssignment) -> Optional[tuple]:
    """IAM-AZ-004: Owner at resource group or resource scope → HIGH."""
    if a.role in HIGH_PRIVILEGE_ROLES and a.scope_type not in ("subscription", "unknown"):
        return (
            "IAM-AZ-004",
            HIGH,
            f"'{a.role}' role at {a.scope_type} scope",
            (
                f"Owner / User Access Administrator grants full control including "
                f"the ability to manage IAM permissions. Assigning '{a.role}' to "
                f"'{a.identity_name}' at {a.scope_type} scope is rarely justified — "
                f"an attacker who controls this identity can escalate privileges "
                f"within the scope."
            ),
            (
                "Replace Owner with Contributor if IAM management is not required, "
                "or use a purpose-built custom role with only the actions needed. "
                "Audit why this role was assigned and remove if not actively used."
            ),
        )
    return None


def _r_unknown_scope_high_priv(a: RoleAssignment) -> Optional[tuple]:
    """IAM-AZ-006: Unresolvable scope for a high-privilege role → MEDIUM."""
    high_roles = HIGH_PRIVILEGE_ROLES | {"Contributor"}
    if a.role in high_roles and a.scope_type == "unknown":
        # Already caught by IAM-AZ-001/002 for subscription inferred unknown
        # This catches cases where scope is truly opaque (var.* reference)
        return (
            "IAM-AZ-006",
            MEDIUM,
            f"'{a.role}' role with unresolvable scope",
            (
                f"The scope of the '{a.role}' assignment for '{a.identity_name}' "
                f"cannot be statically determined (value: '{a.scope_value}'). "
                f"If this resolves to subscription or resource-group level, it "
                f"represents a significant over-permission risk."
            ),
            (
                "Ensure the scope variable resolves to the narrowest possible scope. "
                "Prefer hardcoded resource ARN / resource IDs over variable references "
                "to make IAM scope auditable."
            ),
        )
    return None


# Ordered list of rule functions (first match wins per assignment)
_RULE_FNS = [
    _r_owner_at_subscription,
    _r_contributor_at_subscription,
    _r_contributor_at_rg,
    _r_owner_at_other_scope,
    _r_unknown_scope_high_priv,
]


# ---------------------------------------------------------------------------
# PermissionAnalyzer
# ---------------------------------------------------------------------------


class PermissionAnalyzer:
    """
    Apply the IAM rule set to a list of :class:`~iam_parser.RoleAssignment`
    objects and return :class:`IAMFinding` results.

    Parameters
    ----------
    max_roles_per_identity :
        Number of distinct role assignments above which an identity is
        flagged for having excessive permissions (rule IAM-AZ-005).
    recommendation_engine :
        Custom :class:`RecommendationEngine` instance.  A default one is
        created if not supplied.
    """

    def __init__(
        self,
        max_roles_per_identity: int = 3,
        recommendation_engine: Optional[RecommendationEngine] = None,
    ) -> None:
        self.max_roles_per_identity = max_roles_per_identity
        self.rec = recommendation_engine or RecommendationEngine()

    def analyze(self, assignments: List[RoleAssignment]) -> List[IAMFinding]:
        """
        Run all rules against *assignments* and return deduplicated findings.

        Rules are applied in priority order; at most one rule fires per
        assignment (``break`` after first match).  The multi-role rule is
        evaluated after all single-assignment checks.
        """
        findings: List[IAMFinding] = []

        # Per-assignment rule checks
        for a in assignments:
            for rule_fn in _RULE_FNS:
                result = rule_fn(a)
                if result is None:
                    continue
                rule_id, severity, issue, explanation, fix = result
                recs = self.rec.suggest_roles(a.role, a.identity_name)
                findings.append(
                    IAMFinding(
                        rule_id       = rule_id,
                        identity      = a.identity_name,
                        role          = a.role,
                        scope         = a.scope_type,
                        scope_value   = a.scope_value,
                        severity      = severity,
                        issue         = issue,
                        explanation   = explanation,
                        fix           = fix,
                        recommendations = recs,
                        file_path     = a.file_path,
                        block_name    = a.block_name,
                    )
                )
                break  # highest-priority rule per assignment

        # Multi-role check
        findings.extend(self._check_multiple_roles(assignments))
        findings.extend(self._check_overlapping_roles(assignments))
        findings.extend(self._check_broad_scope_redundancy(assignments))
        findings.extend(self._check_broad_service_access(assignments))

        # Sort: HIGH → MEDIUM → LOW
        _order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        findings.sort(key=lambda f: _order.get(f.severity, 3))
        return findings

    def _check_multiple_roles(
        self,
        assignments: List[RoleAssignment],
    ) -> List[IAMFinding]:
        """Flag identities with more than *max_roles_per_identity* assignments."""
        by_identity: Dict[str, List[RoleAssignment]] = defaultdict(list)
        for a in assignments:
            by_identity[a.identity_name].append(a)

        findings: List[IAMFinding] = []
        for identity, group in by_identity.items():
            if len(group) <= self.max_roles_per_identity:
                continue
            roles_str = ", ".join(a.role for a in group)
            findings.append(
                IAMFinding(
                    rule_id       = "IAM-AZ-005",
                    identity      = identity,
                    role          = roles_str,
                    scope         = group[0].scope_type,
                    scope_value   = "",
                    severity      = MEDIUM,
                    issue         = (
                        f"Identity '{identity}' has {len(group)} role assignments "
                        f"(threshold: {self.max_roles_per_identity})"
                    ),
                    explanation   = (
                        f"Multiple role assignments ({roles_str}) increase the blast "
                        f"radius if this identity is compromised. Overlapping or "
                        f"redundant roles also make auditing harder."
                    ),
                    fix           = (
                        "Review all role assignments for this identity. Remove "
                        "redundant or overlapping roles. Consolidate into a single, "
                        "appropriately scoped role or a minimal custom role."
                    ),
                    recommendations = ["Audit and consolidate role assignments"],
                    file_path       = group[0].file_path,
                    block_name      = "",
                )
            )
        return findings

    def _check_overlapping_roles(
        self,
        assignments: List[RoleAssignment],
    ) -> List[IAMFinding]:
        by_identity: Dict[str, List[RoleAssignment]] = defaultdict(list)
        for assignment in assignments:
            by_identity[assignment.identity_name].append(assignment)

        findings: List[IAMFinding] = []
        seen_pairs: set = set()
        for identity, group in by_identity.items():
            for index, left in enumerate(group):
                for right in group[index + 1 :]:
                    same_scope = bool(left.scope_value and left.scope_value == right.scope_value)
                    nested_scope = (
                        _is_scope_broader(left.scope_type, right.scope_type)
                        and _scope_contains(left.scope_value, right.scope_value)
                    ) or (
                        _is_scope_broader(right.scope_type, left.scope_type)
                        and _scope_contains(right.scope_value, left.scope_value)
                    )
                    if not (same_scope or nested_scope):
                        continue

                    if _role_supersedes(left.role, right.role):
                        dominant, redundant = left, right
                    elif _role_supersedes(right.role, left.role):
                        dominant, redundant = right, left
                    else:
                        continue

                    pair_key = (
                        identity,
                        dominant.role,
                        redundant.role,
                        dominant.scope_value or redundant.scope_value,
                    )
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    findings.append(
                        IAMFinding(
                            rule_id="IAM-AZ-007",
                            identity=identity,
                            role=f"{dominant.role} + {redundant.role}",
                            scope=dominant.scope_type,
                            scope_value=dominant.scope_value or redundant.scope_value,
                            severity=MEDIUM,
                            issue="Overlapping role assignments for one identity",
                            explanation=(
                                f"Identity '{identity}' has overlapping permissions via "
                                f"'{dominant.role}' and '{redundant.role}'. The broader "
                                f"role already covers the effective access granted by the "
                                f"narrower assignment, increasing blast radius and making "
                                f"the identity harder to audit."
                            ),
                            fix=(
                                f"Remove the redundant '{redundant.role}' assignment and keep "
                                f"only the minimum role required for the workload. Prefer a "
                                f"single least-privilege assignment at the narrowest scope."
                            ),
                            recommendations=["Audit and remove overlapping role assignments"],
                            file_path=dominant.file_path or redundant.file_path,
                            block_name=dominant.block_name or redundant.block_name,
                        )
                    )
        return findings

    def _check_broad_scope_redundancy(
        self,
        assignments: List[RoleAssignment],
    ) -> List[IAMFinding]:
        by_identity_role: Dict[tuple, List[RoleAssignment]] = defaultdict(list)
        for assignment in assignments:
            by_identity_role[(assignment.identity_name, assignment.role)].append(assignment)

        findings: List[IAMFinding] = []
        for (identity, role), group in by_identity_role.items():
            for broad in group:
                for narrow in group:
                    if broad is narrow or broad.role != narrow.role:
                        continue
                    if not _is_scope_broader(broad.scope_type, narrow.scope_type):
                        continue
                    if not broad.scope_value or not narrow.scope_value:
                        continue
                    if not _scope_contains(broad.scope_value, narrow.scope_value):
                        continue

                    narrowed_scope = self.rec.suggest_scope(broad.scope_type)
                    severity = MEDIUM if broad.scope_type == "subscription" else LOW
                    findings.append(
                        IAMFinding(
                            rule_id="IAM-AZ-008",
                            identity=identity,
                            role=role,
                            scope=broad.scope_type,
                            scope_value=broad.scope_value,
                            severity=severity,
                            issue="Role assigned at broader scope than necessary",
                            explanation=(
                                f"Identity '{identity}' has role '{role}' at {broad.scope_type} "
                                f"scope even though a narrower {narrow.scope_type} assignment "
                                f"already exists. This broad assignment expands the blast radius "
                                f"without adding unique access."
                            ),
                            fix=(
                                f"Remove the broader {broad.scope_type}-level '{role}' assignment "
                                f"and keep access scoped to {narrow.scope_type} or resource level."
                            ),
                            recommendations=[
                                f"Reduce scope from {broad.scope_type} to {narrowed_scope}",
                            ],
                            file_path=broad.file_path,
                            block_name=broad.block_name,
                        )
                    )
                    break
        return findings

    def _check_broad_service_access(
        self,
        assignments: List[RoleAssignment],
    ) -> List[IAMFinding]:
        findings: List[IAMFinding] = []
        for assignment in assignments:
            if assignment.role not in SERVICE_WIDE_ACCESS_ROLES:
                continue
            if assignment.scope_type not in ("subscription", "resource_group"):
                continue

            severity = MEDIUM if assignment.scope_type == "subscription" else LOW
            recommended_scope = self.rec.suggest_scope(assignment.scope_type)
            findings.append(
                IAMFinding(
                    rule_id="IAM-AZ-009",
                    identity=assignment.identity_name,
                    role=assignment.role,
                    scope=assignment.scope_type,
                    scope_value=assignment.scope_value,
                    severity=severity,
                    issue="Broad service-wide access role assignment",
                    explanation=(
                        f"Role '{assignment.role}' grants broad control across an entire Azure "
                        f"service family. Assigned to '{assignment.identity_name}' at "
                        f"{assignment.scope_type} scope, it can affect many resources without "
                        f"resource-level restriction."
                    ),
                    fix=(
                        f"Scope '{assignment.role}' down to the specific resource or resource "
                        f"group the workload manages, or replace it with a narrower built-in "
                        f"role such as {', '.join(self.rec.suggest_roles(assignment.role)[:2])}."
                    ),
                    recommendations=self.rec.suggest_roles(assignment.role),
                    file_path=assignment.file_path,
                    block_name=assignment.block_name,
                )
            )
        return findings
