"""Helpers for resolving canonical project configuration files."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"
DEFAULT_POLICY_PATH = CONFIG_DIR / "policy.yaml"
LEGACY_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LEGACY_POLICY_PATH = PROJECT_ROOT / "policy.yaml"


def resolve_config_path(config_path: str | None = None) -> Path:
    """Resolve a user-supplied config path with support for legacy root files."""
    if config_path and config_path != "config.yaml":
        return Path(config_path)
    if LEGACY_CONFIG_PATH.exists():
        return LEGACY_CONFIG_PATH
    return DEFAULT_CONFIG_PATH


def resolve_policy_path(policy_path: str | None = None) -> Path:
    """Resolve a user-supplied policy path with support for legacy root files."""
    if policy_path and policy_path != "policy.yaml":
        return Path(policy_path)
    if LEGACY_POLICY_PATH.exists():
        return LEGACY_POLICY_PATH
    return DEFAULT_POLICY_PATH