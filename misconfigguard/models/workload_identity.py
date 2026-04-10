"""Workload identity model exports."""

from federation_analyzer import WorkloadIdentityFinding
from misconfigguard.parsing.workload_identity_parser import WorkloadIdentityConfig

__all__ = ["WorkloadIdentityConfig", "WorkloadIdentityFinding"]