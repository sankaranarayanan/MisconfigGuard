"""Exit handling for CI/CD scanner execution."""

from __future__ import annotations


class ExitHandler:
    """Compute process exit codes from aggregated scan results."""

    def exit_code(self, result: dict, fail_on_high: bool = False) -> int:
        summary = result.get("summary", {})
        high_count = int(summary.get("high", 0))
        if fail_on_high and high_count > 0:
            return 1
        return 0