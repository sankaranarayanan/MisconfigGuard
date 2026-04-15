"""
RBACEnforcer — Role-Based Access Control for MisconfigGuard agents and tools.

Provides:
    • ``Role``       — named set of ``Permission`` values
    • ``Permission`` — fine-grained capability enum
    • ``RBACEnforcer`` — central enforcer; checks caller permissions and
                         controls which tools a given role can invoke

Built-in roles
--------------
    VIEWER    — read scan reports only
    ANALYST   — run scans + read reports
    ENGINEER  — run scans, manage indexes, read/write policies
    ADMIN     — all permissions

Tool Access Control
-------------------
Every tool/analyzer name is registered against a minimum ``Permission``.
``RBACEnforcer.can_use_tool(role, tool)`` returns True/False and
``assert_tool_access`` raises ``PermissionError`` if the caller lacks access.

Usage
-----
    enforcer = RBACEnforcer()

    # Check a permission
    enforcer.assert_permission("analyst", Permission.RUN_SCAN)

    # Tool access
    enforcer.assert_tool_access("viewer", "iam_analyzer")   # raises PermissionError

    # Add a custom role
    enforcer.register_role(Role("auditor", {Permission.READ_REPORT, Permission.EXPORT_REPORT}))
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Set


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------

class Permission(str, enum.Enum):
    """Atomic capabilities that can be granted to a role."""

    # Reports / output
    READ_REPORT    = "read_report"
    EXPORT_REPORT  = "export_report"

    # Scanning
    RUN_SCAN       = "run_scan"
    SCHEDULE_SCAN  = "schedule_scan"

    # Index management
    MANAGE_INDEX   = "manage_index"
    DELETE_INDEX   = "delete_index"

    # Policy management
    READ_POLICY    = "read_policy"
    WRITE_POLICY   = "write_policy"

    # Admin
    MANAGE_USERS   = "manage_users"
    MANAGE_ROLES   = "manage_roles"
    VIEW_AUDIT     = "view_audit"

    # Human-in-the-loop
    APPROVE_FINDING = "approve_finding"
    REJECT_FINDING  = "reject_finding"

    # LLM invocation
    INVOKE_LLM     = "invoke_llm"


# ---------------------------------------------------------------------------
# Role
# ---------------------------------------------------------------------------

@dataclass
class Role:
    """A named collection of permissions."""

    name: str
    permissions: Set[Permission] = field(default_factory=set)

    def has(self, permission: Permission) -> bool:
        return permission in self.permissions

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Role) and self.name == other.name


# ---------------------------------------------------------------------------
# Built-in roles
# ---------------------------------------------------------------------------

_BUILT_IN_ROLES: Dict[str, Role] = {
    "viewer": Role(
        name="viewer",
        permissions={
            Permission.READ_REPORT,
        },
    ),
    "analyst": Role(
        name="analyst",
        permissions={
            Permission.READ_REPORT,
            Permission.EXPORT_REPORT,
            Permission.RUN_SCAN,
            Permission.READ_POLICY,
            Permission.INVOKE_LLM,
        },
    ),
    "engineer": Role(
        name="engineer",
        permissions={
            Permission.READ_REPORT,
            Permission.EXPORT_REPORT,
            Permission.RUN_SCAN,
            Permission.SCHEDULE_SCAN,
            Permission.MANAGE_INDEX,
            Permission.READ_POLICY,
            Permission.WRITE_POLICY,
            Permission.INVOKE_LLM,
            Permission.APPROVE_FINDING,
        },
    ),
    "admin": Role(
        name="admin",
        permissions=set(Permission),   # all permissions
    ),
}


# ---------------------------------------------------------------------------
# Tool → minimum required permission
# ---------------------------------------------------------------------------

# Any tool not listed defaults to Permission.RUN_SCAN
_TOOL_PERMISSIONS: Dict[str, Permission] = {
    # Analyzers
    "iam_analyzer":              Permission.RUN_SCAN,
    "workload_identity_analyzer": Permission.RUN_SCAN,
    "prompt_injection_analyzer": Permission.RUN_SCAN,
    "secrets_analyzer":          Permission.RUN_SCAN,
    "permission_analyzer":       Permission.RUN_SCAN,

    # Index / store management
    "vector_store_manager":      Permission.MANAGE_INDEX,
    "rag_pipeline_ingest":       Permission.MANAGE_INDEX,
    "delete_index":              Permission.DELETE_INDEX,

    # Policy
    "policy_engine_write":       Permission.WRITE_POLICY,
    "policy_engine_read":        Permission.READ_POLICY,

    # LLM invocation
    "llm_client":                Permission.INVOKE_LLM,
    "rag_orchestrator":          Permission.INVOKE_LLM,

    # Admin
    "audit_log_export":          Permission.VIEW_AUDIT,
    "user_management":           Permission.MANAGE_USERS,

    # Human-in-the-loop
    "approve_finding":           Permission.APPROVE_FINDING,
    "reject_finding":            Permission.REJECT_FINDING,
}


# ---------------------------------------------------------------------------
# Enforcer
# ---------------------------------------------------------------------------

class RBACEnforcer:
    """
    Central RBAC enforcer.

    Thread-safe (roles dict is read-mostly; writes protected by convention
    — register roles at startup only).
    """

    def __init__(self) -> None:
        # Shallow copy so built-ins are not mutated across instances
        self._roles: Dict[str, Role] = dict(_BUILT_IN_ROLES)
        self._tool_permissions: Dict[str, Permission] = dict(_TOOL_PERMISSIONS)

    # ------------------------------------------------------------------
    # Role management
    # ------------------------------------------------------------------

    def register_role(self, role: Role) -> None:
        """Add or replace a named role."""
        self._roles[role.name.lower()] = role

    def get_role(self, role_name: str) -> Optional[Role]:
        """Look up a role by name (case-insensitive); returns None if absent."""
        return self._roles.get(role_name.lower())

    # ------------------------------------------------------------------
    # Permission checks
    # ------------------------------------------------------------------

    def has_permission(self, role_name: str, permission: Permission) -> bool:
        """Return True if *role_name* carries *permission*."""
        role = self.get_role(role_name)
        if role is None:
            return False
        return role.has(permission)

    def assert_permission(self, role_name: str, permission: Permission) -> None:
        """
        Raise ``PermissionError`` if *role_name* does not have *permission*.
        """
        if not self.has_permission(role_name, permission):
            raise PermissionError(
                f"Role '{role_name}' lacks permission '{permission.value}'"
            )

    # ------------------------------------------------------------------
    # Tool access control
    # ------------------------------------------------------------------

    def register_tool(self, tool_name: str, required_permission: Permission) -> None:
        """Register a tool and the minimum permission required to invoke it."""
        self._tool_permissions[tool_name.lower()] = required_permission

    def can_use_tool(self, role_name: str, tool_name: str) -> bool:
        """Return True if *role_name* is allowed to invoke *tool_name*."""
        required = self._tool_permissions.get(tool_name.lower(), Permission.RUN_SCAN)
        return self.has_permission(role_name, required)

    def assert_tool_access(self, role_name: str, tool_name: str) -> None:
        """
        Raise ``PermissionError`` if *role_name* is not allowed to use *tool_name*.
        """
        required = self._tool_permissions.get(tool_name.lower(), Permission.RUN_SCAN)
        if not self.has_permission(role_name, required):
            raise PermissionError(
                f"Role '{role_name}' is not permitted to use tool '{tool_name}' "
                f"(requires '{required.value}')"
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_permissions(self, role_name: str) -> Set[Permission]:
        """Return the set of permissions held by *role_name*."""
        role = self.get_role(role_name)
        return set(role.permissions) if role else set()

    def accessible_tools(self, role_name: str) -> list[str]:
        """Return all tool names accessible to *role_name*."""
        return [
            name
            for name, perm in self._tool_permissions.items()
            if self.has_permission(role_name, perm)
        ]
