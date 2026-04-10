"""Exit handling for CI/CD scanner execution."""

from __future__ import annotations


class ExitHandler:
    """Compute process exit codes from aggregated scan results."""

    def exit_code(
        self,
        result: dict,
        fail_on_high: bool = False,
        policy_result: dict | None = None,
    ) -> int:
        summary = result.get("summary", {})
        if int(summary.get("errors", 0) or 0) > 0:
            return 2

        if policy_result and policy_result.get("status") == "fail":
            return 1

        high_count = int(summary.get("high", 0))
        if fail_on_high and high_count > 0:
            return 1
        return 0