"""IAM-related model exports."""

from misconfigguard.analysis.permission_analyzer import IAMFinding
from misconfigguard.parsing.iam_parser import RoleAssignment

__all__ = ["IAMFinding", "RoleAssignment"]