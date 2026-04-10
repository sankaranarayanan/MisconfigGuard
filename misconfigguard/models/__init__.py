"""Shared model exports for the canonical package layout."""

from misconfigguard.models.iam import IAMFinding, RoleAssignment
from misconfigguard.models.workload_identity import WorkloadIdentityConfig, WorkloadIdentityFinding

__all__ = [
    "IAMFinding",
    "RoleAssignment",
    "WorkloadIdentityConfig",
    "WorkloadIdentityFinding",
]