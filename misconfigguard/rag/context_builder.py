"""Optimized LLM context assembly for rule-aware retrieval."""

from __future__ import annotations

from typing import Any, List, Optional

from prompt_builder import PromptBuilder


class ContextBuilder:
    """Wrap PromptBuilder with an optional matched-resource section."""

    def __init__(self, prompt_builder: Optional[PromptBuilder] = None) -> None:
        self.prompt_builder = prompt_builder or PromptBuilder()

    def build(
        self,
        query: str,
        code_results: List[Any],
        security_results: List[Any],
        matched_resources: Optional[List[dict]] = None,
        intent_hint: str = "",
    ) -> str:
        prompt = self.prompt_builder.build(
            query=query,
            code_results=code_results,
            security_results=security_results,
        )
        if intent_hint:
            marker = "## Analysis Query"
            intent_line = f"Focus your analysis on: {intent_hint} security issues.\n\n"
            if marker in prompt:
                prompt = prompt.replace(marker, intent_line + marker, 1)
            else:
                prompt = intent_line + prompt
        if not matched_resources:
            return prompt

        resource_lines = [
            f"- resource_type={resource.get('resource_type', '')}, cloud_provider={resource.get('cloud_provider', '')}, category={resource.get('category', '')}"
            for resource in matched_resources
        ]
        resource_section = "## Matched Resource Tags\n" + "\n".join(resource_lines) + "\n\n"
        marker = "## Code Context (IaC Snippets)"
        if marker in prompt:
            return prompt.replace(marker, resource_section + marker, 1)
        task_marker = "## Task"
        if task_marker in prompt:
            return prompt.replace(task_marker, resource_section + task_marker, 1)
        return resource_section + prompt