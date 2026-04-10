"""CLI entry point for CI/CD-focused repository scanning."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from cli import load_config, setup_logging
from exit_handler import ExitHandler
from pipeline_runner import PipelineRunner
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
    return parser


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

    reports = ReportGenerator()
    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(reports.render_table(result))

    if args.output:
        reports.write_json(result, args.output)
    if args.sarif_output:
        reports.write_sarif(result, args.sarif_output)

    if int(result.get("summary", {}).get("errors", 0) or 0) > 0:
        return 1

    return ExitHandler().exit_code(result, fail_on_high=args.fail_on_high)


if __name__ == "__main__":
    raise SystemExit(main())