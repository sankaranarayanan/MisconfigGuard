"""Hybrid prompt-injection detection for CI/CD pipeline configurations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from injection_detector import InjectionDetector
from input_trust_analyzer import InputTrustAnalyzer
from llm_validator import LLMValidator
from pipeline_config_parser import PipelineConfigParser, PipelineSnippet
from script_analyzer import ScriptAnalyzer

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".yaml", ".yml"}

_FIX_BY_TYPE = {
    "prompt_injection": "Sanitize pipeline-supplied text, avoid embedding untrusted instructions into LLM prompts, and keep system prompts immutable.",
    "script_injection": "Remove dynamic or remote script execution, pin trusted tooling, and avoid eval-style execution of constructed commands.",
    "external_input": "Validate and sanitize external inputs before using them in shell commands or AI prompts, and treat PR metadata as untrusted input.",
}


class PromptInjectionAnalyzer:
    """Detect prompt-injection and command-injection risks in CI/CD YAML."""

    def __init__(
        self,
        pipeline: Optional[Any] = None,
        use_llm: bool = True,
        top_k_security: int = 3,
    ) -> None:
        self.parser = PipelineConfigParser()
        self.injection_detector = InjectionDetector()
        self.script_analyzer = ScriptAnalyzer()
        self.input_trust_analyzer = InputTrustAnalyzer()
        self.llm_validator = LLMValidator(pipeline=pipeline, top_k_security=top_k_security)
        self.pipeline = pipeline
        self.use_llm = use_llm and pipeline is not None
        self.top_k_security = top_k_security

    def analyze_file(self, file_path: str, query: str = "") -> dict:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(file_path)
        snippets = self.parser.parse_file(str(path))
        return self._run_analysis(
            snippets=snippets,
            query=query or f"Detect prompt injection and unsafe AI workflow manipulation in {file_path}",
            files=[str(path)],
        )

    def analyze_directory(self, directory: str, query: str = "") -> dict:
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(directory)

        snippets: List[PipelineSnippet] = []
        files: List[str] = []
        for path in root.rglob("*"):
            if path.suffix.lower() not in _SUPPORTED_SUFFIXES or not path.is_file():
                continue
            parsed = self.parser.parse_file(str(path))
            if parsed:
                files.append(str(path))
                snippets.extend(parsed)

        return self._run_analysis(
            snippets=snippets,
            query=query or f"Detect prompt injection and unsafe AI workflow manipulation in {directory}",
            files=files,
        )

    def analyze_chunks(self, chunks: List[dict], query: str = "Detect prompt injection and unsafe AI workflow manipulation") -> dict:
        snippets: List[PipelineSnippet] = []
        files: List[str] = []
        for index, chunk in enumerate(chunks):
            payload = chunk.get("chunk", chunk)
            text = payload.get("text") or payload.get("content") or ""
            file_path = payload.get("file_path") or f"retrieved_pipeline_{index}.yaml"
            if not text:
                continue
            parsed = self._parse_chunk_text(text, file_path)
            if parsed:
                snippets.extend(parsed)
                files.append(file_path)

        return self._run_analysis(
            snippets=snippets,
            query=query,
            files=files,
        )

    def _parse_chunk_text(self, text: str, file_path: str) -> List[PipelineSnippet]:
        path = Path(file_path)
        try:
            document = yaml.safe_load(text)
        except yaml.YAMLError:
            document = None

        if document is not None and self.parser._looks_like_pipeline(document, path):
            snippets: List[PipelineSnippet] = []
            for step in self.parser._collect_steps(document):
                if not isinstance(step, dict):
                    continue
                step_name = str(
                    step.get("name")
                    or step.get("displayName")
                    or step.get("task")
                    or step.get("uses")
                    or "pipeline step"
                )
                serialized = json.dumps(step, ensure_ascii=True, default=str)
                for source in self.parser._SCRIPT_KEYS:
                    script = step.get(source)
                    if not isinstance(script, str) or not script.strip():
                        continue
                    snippets.append(
                        PipelineSnippet(
                            file_path=file_path,
                            name=step_name,
                            script=script,
                            serialized=serialized,
                            line_number=self.parser._line_for_script(text, script),
                            source=source,
                        )
                    )
            if snippets:
                return snippets

        return self.parser._parse_text_fallback(text, path)

    def _run_analysis(self, snippets: List[PipelineSnippet], query: str, files: List[str]) -> dict:
        findings = self._scan_snippets(snippets)
        llm_response = ""
        llm_issues: List[Dict[str, Any]] = []
        security_refs: List[dict] = []

        if self.use_llm and snippets and findings:
            try:
                combined = "\n\n".join(f"# {snippet.name}\n{snippet.script}" for snippet in snippets[:10])
                llm_response, llm_issues, security_refs = self.llm_validator.validate(
                    file_path=snippets[0].file_path,
                    snippet=combined,
                    findings=findings,
                    query=query,
                )
            except Exception as exc:
                logger.warning("Prompt injection LLM validation failed: %s", exc)

        issues = self._merge_findings(findings, llm_issues)
        summary = self._build_summary(issues, files)
        return {
            "issues": issues,
            "summary": summary,
            "evidence": {
                "security_references": security_refs,
            },
            "analysis": llm_response,
            "metadata": {
                "files_analyzed": sorted(set(files)),
                "scan_findings_count": len(findings),
                "llm_used": bool(llm_response),
            },
        }

    def _scan_snippets(self, snippets: List[PipelineSnippet]) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        seen = set()
        for snippet in snippets:
            candidates = []
            candidates.extend(self.injection_detector.scan_text(snippet.serialized, file_path=snippet.file_path, base_line=snippet.line_number))
            candidates.extend(self.script_analyzer.scan_text(snippet.script, file_path=snippet.file_path, base_line=snippet.line_number))
            candidates.extend(self.input_trust_analyzer.scan_text(snippet.serialized, file_path=snippet.file_path, base_line=snippet.line_number))

            for finding in candidates:
                key = (
                    finding.get("type", ""),
                    finding.get("file_path", ""),
                    finding.get("line_number", 0),
                    finding.get("issue", ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                finding.setdefault("fix", _FIX_BY_TYPE.get(finding.get("type", ""), "Remove the risky pattern and validate all untrusted input before execution."))
                findings.append(finding)
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        findings.sort(key=lambda item: (order.get(item.get("severity", "LOW"), 9), item.get("file_path", ""), item.get("line_number", 0)))
        return findings

    def _merge_findings(self, findings: List[Dict[str, Any]], llm_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged = list(findings)
        seen = {
            (item.get("type", ""), item.get("file_path", ""), item.get("line_number", 0), item.get("issue", ""))
            for item in merged
        }
        for issue in llm_issues:
            key = (
                issue.get("type", ""),
                issue.get("file_path", ""),
                issue.get("line_number", 0),
                issue.get("issue", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            issue.setdefault("confidence", "medium")
            issue.setdefault("rule_id", "PIPE-LLM")
            issue.setdefault("fix", _FIX_BY_TYPE.get(issue.get("type", ""), "Remove the risky pattern and validate all untrusted input before execution."))
            merged.append(issue)
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        merged.sort(key=lambda item: (order.get(item.get("severity", "LOW"), 9), item.get("file_path", ""), item.get("line_number", 0)))
        return merged

    def _build_summary(self, issues: List[Dict[str, Any]], files: List[str]) -> Dict[str, Any]:
        return {
            "total_findings": len(issues),
            "high_severity": sum(1 for issue in issues if issue.get("severity") == "HIGH"),
            "medium_severity": sum(1 for issue in issues if issue.get("severity") == "MEDIUM"),
            "low_severity": sum(1 for issue in issues if issue.get("severity") == "LOW"),
            "files_analyzed": sorted(set(files)),
        }