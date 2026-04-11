"""Helpers for resolving canonical project configuration files."""

from __future__ import annotations

from pathlib import Path

import yaml

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


def load_project_config(config_path: str | None = None) -> dict:
    """Load the canonical project config file, returning an empty dict on failure."""
    resolved_path = resolve_config_path(config_path)
    try:
        with open(resolved_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return {}


def load_llm_config(config_path: str | None = None) -> dict:
    """Return the configured LLM settings from project config."""
    cfg = load_project_config(config_path)
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    return llm_cfg if isinstance(llm_cfg, dict) else {}