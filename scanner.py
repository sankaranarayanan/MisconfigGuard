"""CLI entry point for CI/CD-focused repository scanning."""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

from cli import load_config, setup_logging
from exit_handler import ExitHandler
from pipeline_runner import PipelineRunner
from policy_engine import PolicyEngine
from report_generator import ReportGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scanner.py",
        description="Run MisconfigGuard detectors in CI/CD pipelines",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--path", required=True, help="Repository path or single file to scan")
    parser.add_argument("--output", default=None, help="Write JSON results to this file")
    parser.add_argument("--sarif-output", default=None, dest="sarif_output", help="Write SARIF results to this file")
    parser.add_argument("--fail-on-high", action="store_true", dest="fail_on_high", help="Exit non-zero when HIGH severity findings are present")
    parser.add_argument("--format", choices=["json", "table"], default="table", help="Console output format")
    parser.add_argument("--changed-file", action="append", default=[], dest="changed_files", help="Relative or absolute changed file path to limit scanning; may be repeated")
    parser.add_argument("--max-workers", type=int, default=None, dest="max_workers", help="Maximum parallel workers for file scanning")
    parser.add_argument("--no-llm", action="store_true", dest="no_llm", help="Disable LLM enrichment and run deterministic checks only")
    parser.add_argument("--policy", default=None, help="Path to YAML or JSON policy file")
    parser.add_argument("--policy-env", default=None, dest="policy_env", help="Policy environment override, e.g. dev or prod")
    parser.add_argument("--fail-high", action="store_true", dest="fail_high", help="Override policy to fail on any HIGH severity findings")
    parser.add_argument("--fail-medium", action="store_true", dest="fail_medium", help="Override policy to fail on any MEDIUM severity findings")
    parser.add_argument("--fail-low", action="store_true", dest="fail_low", help="Override policy to fail on any LOW severity findings")
    parser.add_argument("--max-high", type=int, default=None, dest="max_high", help="Maximum allowed HIGH severity findings")
    parser.add_argument("--max-medium", type=int, default=None, dest="max_medium", help="Maximum allowed MEDIUM severity findings")
    parser.add_argument("--max-low", type=int, default=None, dest="max_low", help="Maximum allowed LOW severity findings")
    return parser


def _policy_overrides_from_args(args: argparse.Namespace) -> dict:
    overrides: dict = {"fail_on": {}, "max_allowed": {}}
    for severity in ("high", "medium", "low"):
        fail_value = getattr(args, f"fail_{severity}", False)
        max_value = getattr(args, f"max_{severity}", None)
        if fail_value:
            overrides["fail_on"][severity] = True
        if max_value is not None:
            overrides["max_allowed"][severity] = max_value

    if args.fail_on_high:
        overrides["fail_on"]["high"] = True

    if not overrides["fail_on"] and not overrides["max_allowed"]:
        return {}
    return overrides


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log_cfg = cfg.get("logging", {})
    setup_logging(level=args.log_level or log_cfg.get("level", "INFO"), log_file=log_cfg.get("file"))

    runner = PipelineRunner(
        cfg=cfg,
        use_llm=not args.no_llm,
        max_workers=args.max_workers,
    )
    result = runner.run(args.path, changed_files=args.changed_files)

    policy_config = cfg.get("policy", {})
    policy_path = args.policy or policy_config.get("path")
    policy_environment = args.policy_env or os.getenv("MISCONFIGGUARD_POLICY_ENV") or policy_config.get("environment")
    try:
        policy_result = PolicyEngine().evaluate(
            result,
            policy_path=policy_path,
            policy=policy_config,
            environment=policy_environment,
            overrides=_policy_overrides_from_args(args),
        )
    except Exception as exc:
        summary = result.setdefault("summary", {})
        summary["errors"] = int(summary.get("errors", 0) or 0) + 1
        metadata = result.setdefault("metadata", {})
        errors = metadata.setdefault("errors", [])
        errors.append({
            "detector": "PolicyEngine",
            "file_path": policy_path or "<default policy>",
            "error": str(exc),
        })
        policy_result = {
            "status": "not_evaluated",
            "violations": [],
            "summary": {
                "high": int(summary.get("high", 0) or 0),
                "medium": int(summary.get("medium", 0) or 0),
                "low": int(summary.get("low", 0) or 0),
            },
            "details": [],
        }
    result["policy"] = policy_result

    reports = ReportGenerator()
    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(reports.render_table(result, policy_result=policy_result))

    if args.output:
        reports.write_json(result, args.output)
    if args.sarif_output:
        reports.write_sarif(result, args.sarif_output)

    return ExitHandler().exit_code(result, fail_on_high=args.fail_on_high, policy_result=policy_result)


if __name__ == "__main__":
    raise SystemExit(main())