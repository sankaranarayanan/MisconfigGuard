"""Policy loading for CI/CD enforcement rules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULT_POLICY: Dict[str, Dict[str, Any]] = {
    "fail_on": {
        "high": True,
        "medium": False,
        "low": False,
    },
    "max_allowed": {
        "high": 0,
        "medium": 5,
        "low": 10,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class PolicyLoader:
    """Load policy configuration from YAML/JSON with environment overrides."""

    def load(
        self,
        policy_path: Optional[str] = None,
        *,
        environment: Optional[str] = None,
        base_policy: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        policy = _deep_merge(_DEFAULT_POLICY, base_policy or {})

        if policy_path:
            path = Path(policy_path)
            if not path.is_file():
                raise FileNotFoundError(policy_path)
            raw = path.read_text(encoding="utf-8")
            loaded = self._parse(raw, path.suffix.lower())
            policy = _deep_merge(policy, loaded)

        resolved_environment = environment or os.getenv("MISCONFIGGUARD_POLICY_ENV")
        if resolved_environment:
            envs = policy.get("environments", {})
            if isinstance(envs, dict) and envs and resolved_environment not in envs:
                raise ValueError(f"Unknown policy environment: {resolved_environment}")
            env_policy = envs.get(resolved_environment, {}) if isinstance(envs, dict) else {}
            policy = _deep_merge(policy, env_policy)

        if overrides:
            policy = _deep_merge(policy, overrides)

        policy.pop("environments", None)
        return policy

    @staticmethod
    def _parse(raw: str, suffix: str) -> Dict[str, Any]:
        if suffix == ".json":
            return json.loads(raw)
        return yaml.safe_load(raw) or {}