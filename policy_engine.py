"""End-to-end policy engine for enforcement decisions."""

from __future__ import annotations

from typing import Any, Dict, Optional

from policy_evaluator import PolicyEvaluator
from policy_loader import PolicyLoader


class PolicyEngine:
    """Load policy configuration, apply overrides, and evaluate scan results."""

    def __init__(
        self,
        loader: Optional[PolicyLoader] = None,
        evaluator: Optional[PolicyEvaluator] = None,
    ) -> None:
        self.loader = loader or PolicyLoader()
        self.evaluator = evaluator or PolicyEvaluator()

    def evaluate(
        self,
        result: Dict[str, Any],
        *,
        policy_path: Optional[str] = None,
        policy: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        loaded = self.loader.load(
            policy_path,
            environment=environment,
            base_policy=policy,
            overrides=overrides,
        )
        evaluated = self.evaluator.evaluate(result, loaded)
        evaluated["policy"] = loaded
        return evaluated