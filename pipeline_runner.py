"""CI/CD pipeline runner that aggregates all supported MisconfigGuard analyzers."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from file_scanner import FileScanner

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Run all configured security analyzers over a repository or changed-file set."""

    def __init__(
        self,
        cfg: Optional[dict] = None,
        analyzers: Optional[Sequence[Any]] = None,
        use_llm: bool = False,
        max_workers: Optional[int] = None,
    ) -> None:
        self.cfg = cfg or {}
        ingestion_cfg = self.cfg.get("ingestion", {})
        self.file_scanner = FileScanner(
            max_file_size_mb=ingestion_cfg.get("max_file_size_mb", 10),
        )
        self.use_llm = use_llm
        self.max_workers = max_workers
        self.analyzers = list(analyzers) if analyzers is not None else self._build_default_analyzers()

    def run(self, path: str, changed_files: Optional[Sequence[str]] = None) -> dict:
        target_files = self._resolve_target_files(path, changed_files)
        issues: List[dict] = []
        errors: List[dict] = []

        if target_files:
            worker_count = self.max_workers or min(4, len(target_files)) or 1
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for batch_issues, batch_errors in executor.map(self._analyze_file, target_files):
                    issues.extend(batch_issues)
                    errors.extend(batch_errors)

        issues = self._deduplicate_issues(issues)
        issues.sort(key=lambda issue: (self._severity_rank(issue.get("severity", "LOW")), issue.get("file_path", ""), issue.get("line_number", 0)))

        summary = {
            "total": len(issues),
            "high": sum(1 for issue in issues if issue.get("severity") == "HIGH"),
            "medium": sum(1 for issue in issues if issue.get("severity") == "MEDIUM"),
            "low": sum(1 for issue in issues if issue.get("severity") == "LOW"),
            "errors": len(errors),
            "files_scanned": len(target_files),
            "scanned_files": target_files,
        }

        return {
            "summary": summary,
            "issues": issues,
            "metadata": {
                "changed_files": list(changed_files or []),
                "detectors": [analyzer.__class__.__name__ for analyzer in self.analyzers],
                "errors": errors,
            },
        }

    def _build_default_analyzers(self) -> List[Any]:
        from cli import build_pipeline
        from iam_analyzer import IAMSecurityAnalyzer
        from prompt_injection_analyzer import PromptInjectionAnalyzer
        from secrets_analyzer import HardcodedSecretsAnalyzer
        from workload_identity_analyzer import WorkloadIdentitySecurityAnalyzer

        iam_cfg = self.cfg.get("iam", {})
        prompt_cfg = self.cfg.get("prompt_injection", {})
        wi_cfg = self.cfg.get("workload_identity", {})
        secrets_cfg = self.cfg.get("secrets", {})

        pipeline = None
        if self.use_llm:
            try:
                pipeline = build_pipeline(self.cfg)
                pipeline.load_index()
            except Exception as exc:
                logger.info("PipelineRunner: disabling LLM enrichment in CI: %s", exc)
                pipeline = None

        return [
            IAMSecurityAnalyzer(
                pipeline=pipeline,
                use_llm=self.use_llm,
                max_roles_per_identity=iam_cfg.get("max_roles_per_identity", 3),
                top_k_code=iam_cfg.get("top_k_code", 5),
                top_k_security=iam_cfg.get("top_k_security", 3),
            ),
            WorkloadIdentitySecurityAnalyzer(
                pipeline=pipeline,
                use_llm=self.use_llm,
                top_k_code=wi_cfg.get("top_k_code", 5),
                top_k_security=wi_cfg.get("top_k_security", 3),
            ),
            PromptInjectionAnalyzer(
                pipeline=pipeline,
                use_llm=self.use_llm and prompt_cfg.get("use_llm", True),
                top_k_security=prompt_cfg.get("top_k_security", 3),
            ),
            HardcodedSecretsAnalyzer(
                pipeline=pipeline,
                use_llm=self.use_llm,
                top_k_code=secrets_cfg.get("top_k_code", 5),
                top_k_security=secrets_cfg.get("top_k_security", 3),
                entropy_threshold=secrets_cfg.get("entropy_threshold", 4.5),
            ),
        ]

    def _resolve_target_files(self, path: str, changed_files: Optional[Sequence[str]]) -> List[str]:
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(path)
        if root.is_file():
            return [str(root)]

        if changed_files:
            resolved: List[str] = []
            for candidate in changed_files:
                current = Path(candidate)
                if not current.is_absolute():
                    current = root / candidate
                if current.is_file() and current.suffix.lower() in self.file_scanner.supported_extensions:
                    resolved.append(str(current))
            return sorted(dict.fromkeys(resolved))

        return [str(file_path) for file_path in self.file_scanner.scan(str(root))]

    def _analyze_file(self, file_path: str) -> tuple[List[dict], List[dict]]:
        issues: List[dict] = []
        errors: List[dict] = []
        for analyzer in self.analyzers:
            try:
                result = analyzer.analyze_file(file_path)
            except Exception as exc:
                logger.warning("PipelineRunner: analyzer %s failed for %s: %s", analyzer.__class__.__name__, file_path, exc)
                errors.append({
                    "detector": analyzer.__class__.__name__,
                    "file_path": file_path,
                    "error": str(exc),
                })
                continue
            detector = analyzer.__class__.__name__
            for issue in result.get("issues", []):
                issues.append(self._normalize_issue(issue, file_path, detector))
        return issues, errors

    @staticmethod
    def _normalize_issue(issue: dict, file_path: str, detector: str) -> dict:
        normalized = dict(issue)
        normalized["file_path"] = normalized.get("file_path") or file_path
        normalized["severity"] = (normalized.get("severity") or "LOW").upper()
        normalized["rule_id"] = normalized.get("rule_id") or detector.upper()
        normalized["line_number"] = int(normalized.get("line_number", 0) or 0)
        normalized["detector"] = detector
        return normalized

    @staticmethod
    def _deduplicate_issues(issues: Iterable[dict]) -> List[dict]:
        deduped: List[dict] = []
        seen = set()
        for issue in issues:
            key = (
                issue.get("file_path", ""),
                issue.get("rule_id", ""),
                issue.get("issue", ""),
                issue.get("line_number", 0),
                issue.get("severity", "LOW"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(issue)
        return deduped

    @staticmethod
    def _severity_rank(severity: str) -> int:
        return {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get((severity or "LOW").upper(), 9)