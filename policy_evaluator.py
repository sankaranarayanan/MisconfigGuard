"""Policy evaluation for aggregated security scan results."""

from __future__ import annotations

from typing import Any, Dict, List

_SEVERITIES = ("critical", "high", "medium", "low")


class PolicyEvaluator:
    """Evaluate aggregated findings against an enforcement policy."""

    def evaluate(self, result: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
        summary = result.get("summary", {})
        issues = result.get("issues", [])
        violations: List[str] = []
        details: List[Dict[str, Any]] = []

        for severity in _SEVERITIES:
            count = int(summary.get(severity, 0) or 0)
            max_allowed = int(policy.get("max_allowed", {}).get(severity, 0 if severity in ("critical", "high") else 999999) or 0)
            fail_on = bool(policy.get("fail_on", {}).get(severity, False))

            if count <= 0:
                continue

            if fail_on:
                violations.append(
                    f"{severity.upper()} severity issues found ({count}); policy requires zero tolerance for {severity.upper()}."
                )
                details.append(self._violation_detail(result, severity.upper(), count, max_allowed, issues))
                continue

            if count > max_allowed:
                violations.append(
                    f"{severity.upper()} severity issues exceed allowed threshold ({count} found, max allowed: {max_allowed})."
                )
                details.append(self._violation_detail(result, severity.upper(), count, max_allowed, issues))

        return {
            "status": "fail" if violations else "pass",
            "violations": violations,
            "summary": {
                "critical": int(summary.get("critical", 0) or 0),
                "high": int(summary.get("high", 0) or 0),
                "medium": int(summary.get("medium", 0) or 0),
                "low": int(summary.get("low", 0) or 0),
            },
            "details": details,
        }

    @staticmethod
    def _violation_detail(result: Dict[str, Any], severity: str, count: int, max_allowed: int, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        affected = [
            issue.get("file_path", "")
            for issue in issues
            if issue.get("severity", "").upper() == severity and issue.get("file_path")
        ]
        return {
            "severity": severity,
            "count": count,
            "max_allowed": max_allowed,
            "affected_files": sorted(dict.fromkeys(affected)),
        }