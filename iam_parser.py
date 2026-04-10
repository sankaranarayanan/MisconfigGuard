"""
IAMParser — Extract Azure managed identity and role assignment configurations
from Terraform HCL, YAML ARM templates, and JSON ARM templates.

Produces normalised :class:`RoleAssignment` objects consumed by
:class:`~permission_analyzer.PermissionAnalyzer` and
:class:`~iam_analyzer.IAMSecurityAnalyzer`.

Supported sources
-----------------
Terraform HCL:
    azurerm_role_assignment        — primary target (role, scope, principal)
    azurerm_user_assigned_identity — identity definitions (name tracking)

ARM templates (JSON or YAML):
    Microsoft.Authorization/roleAssignments
    Microsoft.ManagedIdentity/userAssignedIdentities

Pipeline integration:
    parse_chunks() accepts pre-chunked dicts from IntelligentChunker,
    filtering for chunks whose resource_type is azurerm_role_assignment.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Azure built-in role GUID → display name
# ---------------------------------------------------------------------------

AZURE_ROLE_GUIDS: Dict[str, str] = {
    "8e3af657-a8ff-443c-a75c-2fe8c4bcb635": "Owner",
    "b24988ac-6180-42a0-ab88-20f7382dd24c": "Contributor",
    "acdd72a7-3385-48ef-bd42-f606fba81ae7": "Reader",
    "ba92f5b4-2d11-453d-a403-e96b0029c9fe": "Storage Blob Data Contributor",
    "2a2b9908-6ea1-4ae2-8e65-a410df84e7d1": "Storage Blob Data Reader",
    "974c5e8b-45b9-4653-ba55-5f855dd0fb88": "Storage Queue Data Contributor",
    "9980e02c-c2be-4d73-94e8-173b1dc7cf3c": "Virtual Machine Contributor",
    "fb879df8-f326-4884-b1cf-06f3ad86be52": "Virtual Machine Administrator Login",
    "3913510d-42f4-4e42-8a64-420c390055eb": "Virtual Machine User Login",
    "992aa8b3-b645-451c-88c6-62fd8b5a73af": "Key Vault Secrets User",
    "4633458b-17de-408a-b874-0445c86b69e6": "Key Vault Secrets Officer",
    "21090545-7ca7-4776-b22c-e363652d74d2": "Key Vault Reader",
    "00c29273-979b-4161-815c-10b084fb9324": "Key Vault Certificates Officer",
    "18d7d88d-d35e-4fb5-a5c3-7773c20a72d9": "User Access Administrator",
    "749f88d5-cbae-40b8-bcfc-e573ddc772fa": "Monitoring Contributor",
    "92aaf0da-9dab-42b6-933f-5effd83a5c5c": "Log Analytics Reader",
    "3913510d-42f4-4e42-8a64-420c390055eb": "Monitoring Metrics Publisher",
    "b24988ac-6180-42a0-ab88-20f7382dd24c": "Contributor",  # repeated for clarity
}


# ---------------------------------------------------------------------------
# RoleAssignment dataclass
# ---------------------------------------------------------------------------


@dataclass
class RoleAssignment:
    """A single Azure role assignment extracted from an IaC source file."""

    identity_name   : str  # Logical name of the principal/identity
    identity_type   : str  # "user_assigned"|"system_assigned"|"service_principal"|"unknown"
    role            : str  # Role display name (e.g. "Contributor")
    scope_type      : str  # "subscription"|"resource_group"|"resource"|"unknown"
    scope_value     : str  # Raw scope expression from the source
    principal_id_ref: str  # Principal ID reference expression (may be a TF ref)
    file_path       : str  # Source file path
    block_name      : str  # Terraform block label or ARM resource name
    source_text     : str  # Raw block text (≤ 800 chars, for LLM context)

    def to_dict(self) -> dict:
        return {
            "identity_name":    self.identity_name,
            "identity_type":    self.identity_type,
            "role":             self.role,
            "scope_type":       self.scope_type,
            "scope_value":      self.scope_value,
            "principal_id_ref": self.principal_id_ref,
            "file_path":        self.file_path,
            "block_name":       self.block_name,
        }


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def resolve_role_name(value: str) -> str:
    """
    Return a human-readable role name from either a display name or a GUID.

    ARM templates embed role GUIDs inside expressions like::

        subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'GUID')

    This function extracts the GUID and maps it to the display name.
    If the GUID is unrecognised, returns ``"Unknown (<guid>)"``.
    If no GUID pattern is found, strips quotes and returns the raw value.
    """
    if not value:
        return "Unknown"
    # Extract a UUID from the expression
    m = re.search(
        r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
        value,
        re.IGNORECASE,
    )
    if m:
        guid = m.group(1).lower()
        return AZURE_ROLE_GUIDS.get(guid, f"Unknown ({guid})")
    # Plain display name — strip surrounding quotes
    return value.strip().strip("\"'")


def classify_scope(scope_expr: str) -> str:
    """
    Classify a scope expression into one of:
    ``"subscription"``, ``"resource_group"``, ``"resource"``, ``"unknown"``.

    Handles literal paths (``/subscriptions/xxx/resourceGroups/yyy``) and
    Terraform / ARM expression references.
    """
    if not scope_expr:
        return "unknown"

    s  = scope_expr.strip().strip("\"'")
    sl = s.lower()

    # --- Literal path: /subscriptions/... ---
    if sl.startswith("/subscriptions/"):
        parts = [p for p in sl.split("/") if p]
        # [subscriptions, {id}] = 2 → subscription
        # [subscriptions, {id}, resourcegroups, {rg}] = 4 → resource_group
        # more → resource
        if len(parts) <= 2:
            return "subscription"
        if "resourcegroups" in sl or "resourcegroup" in sl:
            return "resource_group" if len(parts) <= 4 else "resource"
        return "resource"

    # --- Terraform / ARM expression heuristics ---
    # Subscription-level indicators
    for hint in (
        "data.azurerm_subscription",
        "azurerm_subscription.",
        ".subscription_id",
        "subscription().id",
        "subscriptionresourceid(",
    ):
        if hint in sl:
            return "subscription"

    # Resource-group-level indicators
    for hint in (
        "azurerm_resource_group.",
        "resourcegroup().id",
        ".resource_group_id",
        "/resourcegroups",
    ):
        if hint in sl:
            return "resource_group"

    # Pure variable / local / parameter reference → can't determine
    if sl.startswith(("var.", "local.", "param")):
        return "unknown"

    # Non-trivial path expression → treat as resource-level
    return "resource" if "/" in s else "unknown"


# ---------------------------------------------------------------------------
# Internal Terraform helpers
# ---------------------------------------------------------------------------


def _extract_tf_resource_blocks(
    content: str,
    resource_type: str,
) -> List[Tuple[str, str, int]]:
    """
    Find all Terraform resource blocks of the given *resource_type*.

    Uses brace-depth tracking to handle arbitrarily nested sub-blocks.

    Returns ``(block_name, block_content, char_offset)`` tuples.
    """
    pattern = re.compile(
        r'resource\s+"' + re.escape(resource_type) + r'"\s+"(\w+)"\s*\{',
        re.MULTILINE | re.IGNORECASE,
    )
    results: List[Tuple[str, str, int]] = []
    for m in pattern.finditer(content):
        block_name = m.group(1)
        start      = m.end()
        depth      = 1
        pos        = start
        while pos < len(content) and depth > 0:
            ch = content[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            pos += 1
        block_content = content[start : pos - 1]
        results.append((block_name, block_content, m.start()))
    return results


def _get_tf_attr(block: str, attr: str) -> Optional[str]:
    """
    Extract the value of a named attribute from a Terraform block snippet.

    Handles:
    * String literals: ``key = "value"``
    * References / expressions: ``key = data.azurerm_subscription.primary.id``
    """
    # String literal
    m = re.search(
        r"^\s*" + re.escape(attr) + r'\s*=\s*"([^"]*)"',
        block,
        re.MULTILINE,
    )
    if m:
        return m.group(1)
    # Reference or expression (not a quoted string, not a block opener)
    m = re.search(
        r"^\s*"
        + re.escape(attr)
        + r"\s*=\s*([^\"#\n{}\[\]][^#\n]*?)(?:\s*#[^\n]*)?\s*$",
        block,
        re.MULTILINE,
    )
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Identity inference helpers
# ---------------------------------------------------------------------------


def _infer_identity_name(pid_ref: str, block_name: str) -> str:
    """
    Derive a human-readable identity label from a principal_id reference.

    Examples::

        azurerm_user_assigned_identity.main.principal_id  → "main"
        azurerm_linux_virtual_machine.web.identity[0].principal_id → "web"
        parameters('identityName')                        → "identityName"
        var.principal_id                                  → block_name (fallback)
    """
    if not pid_ref:
        return block_name
    # azurerm_<type>.<name>.<attribute>
    m = re.search(r"azurerm_\w+\.(\w+)\.", pid_ref)
    if m:
        return m.group(1)
    # ARM parameter reference: parameters('name') or parameters("name")
    m = re.search(r"parameters?\(['\"](\w+)['\"]\)", pid_ref, re.IGNORECASE)
    if m:
        return m.group(1)
    return block_name


def _infer_identity_type(pid_ref: str) -> str:
    """Guess identity type from the principal_id reference string."""
    if not pid_ref:
        return "unknown"
    pl = pid_ref.lower()
    if "user_assigned_identity" in pl or "userassignedidentity" in pl:
        return "user_assigned"
    if "linux_virtual_machine" in pl or "windows_virtual_machine" in pl:
        return "system_assigned"
    if "service_principal" in pl:
        return "service_principal"
    return "unknown"


# ---------------------------------------------------------------------------
# IAMParser
# ---------------------------------------------------------------------------


class IAMParser:
    """
    Parse Azure IAM configurations from Terraform, YAML, and JSON sources.

    All ``parse_*`` methods return a list of :class:`RoleAssignment` objects.
    """

    # ------------------------------------------------------------------
    # Terraform
    # ------------------------------------------------------------------

    def parse_terraform(self, content: str, file_path: str = "") -> List[RoleAssignment]:
        """
        Extract role assignments from Terraform HCL content.

        Targets ``azurerm_role_assignment`` resource blocks.
        """
        results: List[RoleAssignment] = []
        blocks = _extract_tf_resource_blocks(content, "azurerm_role_assignment")

        for block_name, block_text, _ in blocks:
            role_name = _get_tf_attr(block_text, "role_definition_name")
            role_id   = _get_tf_attr(block_text, "role_definition_id")
            scope_raw = _get_tf_attr(block_text, "scope")
            pid_ref   = _get_tf_attr(block_text, "principal_id") or ""

            role = resolve_role_name(role_name or role_id or "")
            if not role or role == "Unknown":
                logger.debug(
                    "Skipping azurerm_role_assignment.%s — no role resolved",
                    block_name,
                )
                continue

            full_text = (
                f'resource "azurerm_role_assignment" "{block_name}" {{\n'
                f"{block_text}\n}}"
            )
            results.append(
                RoleAssignment(
                    identity_name    = _infer_identity_name(pid_ref, block_name),
                    identity_type    = _infer_identity_type(pid_ref),
                    role             = role,
                    scope_type       = classify_scope(scope_raw or ""),
                    scope_value      = scope_raw or "",
                    principal_id_ref = pid_ref,
                    file_path        = file_path,
                    block_name       = block_name,
                    source_text      = full_text[:800],
                )
            )

        logger.debug(
            "IAMParser: %d role assignment(s) in %s",
            len(results),
            file_path or "<string>",
        )
        return results

    # ------------------------------------------------------------------
    # YAML ARM templates
    # ------------------------------------------------------------------

    def parse_yaml(self, content: str, file_path: str = "") -> List[RoleAssignment]:
        """Parse role assignments from a YAML ARM template."""
        try:
            import yaml
            data = yaml.safe_load(content)
        except Exception as exc:
            logger.debug("YAML parse failed (%s): %s", file_path, exc)
            return []
        return self._parse_arm_dict(data, file_path)

    # ------------------------------------------------------------------
    # JSON ARM templates
    # ------------------------------------------------------------------

    def parse_json(self, content: str, file_path: str = "") -> List[RoleAssignment]:
        """Parse role assignments from a JSON ARM template."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.debug("JSON parse failed (%s): %s", file_path, exc)
            return []
        return self._parse_arm_dict(data, file_path)

    # ------------------------------------------------------------------
    # Chunk-based (pipeline integration)
    # ------------------------------------------------------------------

    def parse_chunks(self, chunks: List[dict]) -> List[RoleAssignment]:
        """
        Extract role assignments from pre-chunked data produced by
        :class:`~intelligent_chunker.IntelligentChunker`.

        For Terraform chunks with ``resource_type == "azurerm_role_assignment"``
        the chunk text is re-parsed with :meth:`parse_terraform`.
        For YAML/JSON chunks that mention roleAssignment, the text is
        parsed with :meth:`parse_yaml` / :meth:`parse_json`.
        """
        results: List[RoleAssignment] = []
        for chunk in chunks:
            meta      = chunk.get("metadata", {})
            file_type = chunk.get("file_type", "")
            file_path = chunk.get("file_path", "")
            res_type  = meta.get("resource_type", "")
            text      = chunk.get("text", "")

            if file_type == "terraform" and res_type == "azurerm_role_assignment":
                results.extend(self.parse_terraform(text, file_path))
            elif file_type in ("yaml", "json") and text:
                lowered = text.lower()
                if "roleassignment" in lowered or "roledefinition" in lowered:
                    parser = self.parse_yaml if file_type == "yaml" else self.parse_json
                    results.extend(parser(text, file_path))

        # Deduplicate by (file_path, block_name, role, scope_value)
        seen: set = set()
        deduped: List[RoleAssignment] = []
        for r in results:
            key = (r.file_path, r.block_name, r.role, r.scope_value)
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped

    # ------------------------------------------------------------------
    # File-level convenience
    # ------------------------------------------------------------------

    def parse_file(self, file_path: str) -> List[RoleAssignment]:
        """Auto-detect file type and parse accordingly."""
        path = Path(file_path)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Could not read %s: %s", file_path, exc)
            return []
        suffix = path.suffix.lower()
        if suffix == ".tf":
            return self.parse_terraform(content, file_path)
        if suffix in (".yaml", ".yml"):
            return self.parse_yaml(content, file_path)
        if suffix == ".json":
            return self.parse_json(content, file_path)
        return []

    # ------------------------------------------------------------------
    # ARM template dict traversal
    # ------------------------------------------------------------------

    def _parse_arm_dict(self, data: object, file_path: str) -> List[RoleAssignment]:
        """Recursively locate roleAssignment resources in an ARM data structure."""
        results: List[RoleAssignment] = []
        if isinstance(data, dict):
            self._visit_arm_node(data, file_path, results)
        elif isinstance(data, list):
            for item in data:
                results.extend(self._parse_arm_dict(item, file_path))
        return results

    def _visit_arm_node(
        self,
        node: dict,
        file_path: str,
        results: List[RoleAssignment],
    ) -> None:
        """Visit one dict node in an ARM template tree."""
        rtype = (node.get("type") or node.get("Type") or "").lower()
        if "roleassignment" in rtype:
            ra = self._arm_node_to_assignment(node, file_path)
            if ra:
                results.append(ra)
        # Recurse into child collections
        for key in ("resources", "properties", "parameters", "variables"):
            child = node.get(key)
            if isinstance(child, list):
                for item in child:
                    if isinstance(item, dict):
                        self._visit_arm_node(item, file_path, results)
            elif isinstance(child, dict):
                self._visit_arm_node(child, file_path, results)

    def _arm_node_to_assignment(
        self,
        node: dict,
        file_path: str,
    ) -> Optional[RoleAssignment]:
        """Convert an ARM roleAssignment resource node to :class:`RoleAssignment`."""
        props   = node.get("properties") or {}
        name    = str(node.get("name", ""))
        role_id = str(props.get("roleDefinitionId") or props.get("RoleDefinitionId") or "")
        pid     = str(props.get("principalId") or props.get("PrincipalId") or "")
        scope   = str(props.get("scope") or props.get("Scope") or "")

        role = resolve_role_name(role_id)
        if not role or role.startswith("Unknown"):
            return None

        # Role assignments deployed at subscription scope often omit the scope
        # property — treat as subscription level.
        scope_type = classify_scope(scope) if scope else "subscription"

        return RoleAssignment(
            identity_name    = _infer_identity_name(pid, name),
            identity_type    = "unknown",
            role             = role,
            scope_type       = scope_type,
            scope_value      = scope,
            principal_id_ref = pid,
            file_path        = file_path,
            block_name       = name,
            source_text      = json.dumps(node, indent=2)[:800],
        )
