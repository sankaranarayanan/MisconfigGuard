"""Parse CI/CD pipeline YAML files into script-bearing snippets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import yaml


@dataclass
class PipelineSnippet:
    file_path: str
    name: str
    script: str
    serialized: str
    line_number: int
    source: str


class PipelineConfigParser:
    """Extract runnable pipeline steps and their surrounding inputs from YAML files."""

    _SCRIPT_KEYS = ("run", "script", "bash", "pwsh", "powershell")
    _PIPELINE_TEXT_MARKERS = ("jobs:", "steps:", "stages:", "trigger:", "pr:", "on:")

    def parse_file(self, file_path: str) -> List[PipelineSnippet]:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(file_path)

        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            document = yaml.safe_load(text)
        except yaml.YAMLError:
            document = None

        if document is None:
            return self._parse_text_fallback(text, path)

        if not self._looks_like_pipeline(document, path):
            return []

        snippets: List[PipelineSnippet] = []
        for step in self._collect_steps(document):
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
            for source in self._SCRIPT_KEYS:
                script = step.get(source)
                if not isinstance(script, str) or not script.strip():
                    continue
                snippets.append(
                    PipelineSnippet(
                        file_path=str(path),
                        name=step_name,
                        script=script,
                        serialized=serialized,
                        line_number=self._line_for_script(text, script),
                        source=source,
                    )
                )
        if snippets:
            return snippets
        return self._parse_text_fallback(text, path)

    def _parse_text_fallback(self, text: str, path: Path) -> List[PipelineSnippet]:
        if not self._looks_like_pipeline_text(text, path):
            return []

        snippets: List[PipelineSnippet] = []
        current_name = "pipeline step"
        lines = text.splitlines()

        for index, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            name_match = re.match(r"^-\s*(?:name|displayName)\s*:\s*(.+)$", stripped)
            if name_match:
                current_name = name_match.group(1).strip().strip('"\'') or "pipeline step"
                continue

            script_match = re.match(r"^-?\s*(run|script|bash|pwsh|powershell)\s*:\s*(.*)$", stripped)
            if not script_match:
                continue

            source = script_match.group(1)
            script = self._extract_script_block(lines, index - 1, script_match.group(2), line)
            if not script.strip():
                continue

            snippets.append(
                PipelineSnippet(
                    file_path=str(path),
                    name=current_name,
                    script=script,
                    serialized=json.dumps({"name": current_name, source: script}, ensure_ascii=True),
                    line_number=index,
                    source=source,
                )
            )
        return snippets

    def _looks_like_pipeline(self, document: Any, path: Path) -> bool:
        if not isinstance(document, dict):
            return False

        normalized_keys = {str(key).lower() for key in document.keys()}
        if path.name.lower() == "azure-pipelines.yml":
            return True
        if ".github" in path.parts and "workflows" in path.parts:
            return True
        return bool({"jobs", "steps", "stages", "pr", "trigger"} & normalized_keys)

    def _collect_steps(self, node: Any) -> List[dict]:
        steps: List[dict] = []
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).lower() == "steps" and isinstance(value, list):
                    steps.extend(item for item in value if isinstance(item, dict))
                else:
                    steps.extend(self._collect_steps(value))
        elif isinstance(node, list):
            for item in node:
                steps.extend(self._collect_steps(item))
        return steps

    def _looks_like_pipeline_text(self, text: str, path: Path) -> bool:
        if path.name.lower() == "azure-pipelines.yml":
            return True
        if ".github" in path.parts and "workflows" in path.parts:
            return True
        lowered = text.lower()
        return any(marker in lowered for marker in self._PIPELINE_TEXT_MARKERS)

    def _extract_script_block(self, lines: List[str], index: int, remainder: str, original_line: str) -> str:
        inline_script = remainder.strip()
        if inline_script and inline_script not in {"|", ">", "|-", ">-", "|+", ">+"}:
            return inline_script

        base_indent = len(original_line) - len(original_line.lstrip(" "))
        block: List[str] = []
        for next_line in lines[index + 1 :]:
            if not next_line.strip():
                if block:
                    block.append("")
                continue
            current_indent = len(next_line) - len(next_line.lstrip(" "))
            if current_indent <= base_indent:
                break
            block.append(next_line.strip())
        return "\n".join(block).strip()

    def _line_for_script(self, source_text: str, script: str) -> int:
        marker = next((line.strip() for line in script.splitlines() if line.strip()), script.strip())
        if not marker:
            return 1
        for index, line in enumerate(source_text.splitlines(), 1):
            if marker in line:
                return index
        return 1