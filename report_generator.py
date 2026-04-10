"""Report generation helpers for CI/CD scanner outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _severity_to_sarif_level(severity: str) -> str:
    return {
        "HIGH": "error",
        "MEDIUM": "warning",
        "LOW": "note",
    }.get((severity or "").upper(), "note")


class ReportGenerator:
    """Render and persist scanner results for local and CI consumption."""

    def write_json(self, result: Dict[str, Any], output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def write_sarif(self, result: Dict[str, Any], output_path: str) -> None:
        rules = []
        seen_rules = set()
        sarif_results: List[Dict[str, Any]] = []

        for issue in result.get("issues", []):
            rule_id = issue.get("rule_id") or issue.get("detector") or "MISCONFIGGUARD"
            if rule_id not in seen_rules:
                seen_rules.add(rule_id)
                rules.append({
                    "id": rule_id,
                    "name": issue.get("issue", rule_id),
                    "shortDescription": {"text": issue.get("issue", rule_id)},
                    "properties": {"tags": [issue.get("detector", "scanner")]},
                })

            result_entry: Dict[str, Any] = {
                "ruleId": rule_id,
                "level": _severity_to_sarif_level(issue.get("severity", "LOW")),
                "message": {"text": issue.get("explanation") or issue.get("issue", "Security issue detected")},
            }

            file_path = issue.get("file_path")
            if file_path:
                result_entry["locations"] = [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": file_path.replace('\\', '/')},
                        "region": {"startLine": max(int(issue.get("line_number", 1) or 1), 1)},
                    }
                }]

            sarif_results.append(result_entry)

        payload = {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "MisconfigGuard",
                        "informationUri": "https://example.local/misconfigguard",
                        "rules": rules,
                    }
                },
                "results": sarif_results,
            }],
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def render_table(self, result: Dict[str, Any], max_issues: int = 5) -> str:
        summary = result.get("summary", {})
        issues = result.get("issues", [])
        lines = [
            "MisconfigGuard CI Scan Summary",
            f"Total issues: {summary.get('total', 0)}",
            f"HIGH: {summary.get('high', 0)}  MEDIUM: {summary.get('medium', 0)}  LOW: {summary.get('low', 0)}",
            f"Files scanned: {summary.get('files_scanned', 0)}",
        ]

        if issues:
            lines.append("")
            lines.append("Key issues:")
            for issue in issues[:max_issues]:
                lines.append(
                    f"- [{issue.get('severity', 'LOW')}] {issue.get('issue', 'Issue')}"
                    f" ({issue.get('file_path', 'unknown file')})"
                )
        else:
            lines.append("")
            lines.append("No issues detected.")

        return "\n".join(lines)