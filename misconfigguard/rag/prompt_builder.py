"""
PromptBuilder — Assembles structured prompts for the local LLM.

Produces a two-section prompt:
    1. **Code Context** — retrieved IaC chunks most relevant to the query.
    2. **Security Rules** — retrieved CIS/OWASP/best-practice rules.

The builder respects a configurable token budget so the combined prompt
stays within the LLM's context window.  Token counting uses ``tiktoken``
when available, falling back to a whitespace split word-count estimate.

Output format instruction
-------------------------
The prompt always asks the LLM to return a JSON object:

    {
        "issues": [
            {
                "title":              "...",
                "severity":           "CRITICAL|HIGH|MEDIUM|LOW|INFO",
                "description":        "...",
                "affected_resource":  "...",
                "recommendation":     "..."
            }
        ],
        "summary": "..."
    }

Usage
-----
    builder = PromptBuilder(max_context_tokens=3000)
    prompt  = builder.build(
        query           = "Check Terraform for public S3 risks",
        code_results    = retrieval_results,      # List[RetrievalResult]
        security_results= kb_results,             # List[SecurityRuleResult]
    )
    # → str prompt ready for LocalLLMClient
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Optional tiktoken for precise token counting.
try:
    import tiktoken as _tiktoken
    _TIKTOKEN_ENC = _tiktoken.get_encoding("cl100k_base")
    _TIKTOKEN_OK  = True
except Exception:
    _TIKTOKEN_ENC = None
    _TIKTOKEN_OK  = False


def _count_tokens(text: str) -> int:
    """Return an approximate token count for *text*."""
    if _TIKTOKEN_OK and _TIKTOKEN_ENC is not None:
        return len(_TIKTOKEN_ENC.encode(text))
    # Whitespace split: ~1.3 words per token on average.
    return max(1, int(len(text.split()) * 0.75))


def _truncate(text: str, max_tokens: int) -> str:
    """
    Truncate *text* to at most *max_tokens*, appending ``…`` if clipped.
    """
    if _count_tokens(text) <= max_tokens:
        return text
    if _TIKTOKEN_OK and _TIKTOKEN_ENC is not None:
        ids = _TIKTOKEN_ENC.encode(text)[:max_tokens - 1]
        return _TIKTOKEN_ENC.decode(ids) + "…"
    # Word-based approximation (faster than re-checking token counts).
    words     = text.split()
    keep      = int(max_tokens / 0.75)  # inverse of word→token ratio
    truncated = " ".join(words[:keep])
    return truncated + ("…" if len(words) > keep else "")


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


_SYSTEM_PREAMBLE = """\
You are an expert Infrastructure-as-Code (IaC) security engineer.
Your task is to analyse the provided code context for security misconfigurations \
using the referenced security rules and best practices.

Be precise: only report genuine issues that are evidenced by the code context.
Do NOT fabricate findings that are not supported by the provided snippets.\
"""

_OUTPUT_INSTRUCTION = """\
Return **only** a valid JSON object in this exact schema (no markdown fences, \
no extra commentary):

{
  "issues": [
    {
      "title":             "<short issue title>",
      "severity":          "CRITICAL | HIGH | MEDIUM | LOW | INFO",
      "description":       "<what the issue is and why it matters>",
      "affected_resource": "<Terraform resource / Kubernetes object / file path — REQUIRED for CRITICAL/HIGH>",
      "file_path":         "<exact file path from the context header above, or empty string>",
      "evidence_snippet":  "<the exact problematic line(s) quoted from the code context — REQUIRED for CRITICAL/HIGH>",
      "recommendation":    "<concrete fix with config example if possible>",
      "cwe":               "<CWE-NNN — required for CRITICAL/HIGH, e.g. CWE-732>",
      "owasp":             "<OWASP AXX:YYYY — required for CRITICAL/HIGH, e.g. OWASP A05:2021>"
    }
  ],
  "summary": "<1-3 sentence overall assessment>"
}

Severity calibration (apply strictly):
- CRITICAL: direct exploitation possible with no prerequisites (e.g., unauthenticated public access, plaintext root credentials)
- HIGH:     exploitation requires only low-privilege access or single misconfiguration (e.g., overly permissive RBAC, secret in env var)
- MEDIUM:   hardening gap; exploitable only with additional context (e.g., missing MFA, weak TLS version)
- LOW:      best practice violation with no direct exploit path
- INFO:     informational observation

Rules:
- Only report issues evidenced by the provided code context. Do NOT fabricate findings.
- For CRITICAL and HIGH issues, evidence_snippet, affected_resource, and cwe are mandatory.
- If no issues are found, return exactly: {"issues": [], "summary": "No security issues detected in the provided context."}\
"""


class PromptBuilder:
    """
    Build an LLM-ready prompt with token-budget enforcement.

    Parameters
    ----------
    max_context_tokens :
        Maximum total tokens for the code + security-rules sections combined.
        The preamble, query, and output instructions are added on top.
        Default: 3000.
    max_code_ratio :
        Fraction of the context budget reserved for code chunks.  The
        remainder goes to security rules.  Default: 0.65.
    include_metadata :
        If ``True``, prepend a short header to each code chunk showing its
        chunk_id, file path, cloud provider, and resource type.
    """

    def __init__(
        self,
        max_context_tokens: int   = 3000,
        max_code_ratio:     float = 0.65,
        include_metadata:   bool  = True,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.max_code_ratio     = max_code_ratio
        self.include_metadata   = include_metadata

        self._code_budget     = int(max_context_tokens * max_code_ratio)
        self._security_budget = max_context_tokens - self._code_budget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        query:            str,
        code_results:     List[Any],  # List[RetrievalResult]
        security_results: List[Any],  # List[SecurityRuleResult]
    ) -> str:
        """
        Assemble a structured prompt.

        Parameters
        ----------
        query :
            The user's analysis query.
        code_results :
            Ranked list of ``RetrievalResult`` objects from ``HybridRetriever``.
        security_results :
            Ranked list of ``SecurityRuleResult`` objects from
            ``SecurityKnowledgeBase``.

        Returns
        -------
        str
            Complete prompt string ready for ``LocalLLMClient.generate()``.
        """
        code_section     = self._build_code_section(code_results)
        security_section = self._build_security_section(security_results)

        parts = [
            _SYSTEM_PREAMBLE,
            "",
            "## Analysis Query",
            query,
            "",
        ]

        if code_section:
            parts += ["## Code Context (IaC Snippets)", code_section, ""]

        if security_section:
            parts += ["## Security Rules & Best Practices", security_section, ""]

        parts += ["## Task", _OUTPUT_INSTRUCTION]

        return "\n".join(parts)

    def build_simple(self, query: str, context: str) -> str:
        """
        Lightweight variant: plain context string (e.g. from RAGPipeline).

        Used as a fallback when HybridRetriever / SecurityKB are not available.
        """
        context_trunc = _truncate(context, self.max_context_tokens)
        return "\n\n".join([
            _SYSTEM_PREAMBLE,
            f"## Analysis Query\n{query}",
            f"## Context\n{context_trunc}",
            f"## Task\n{_OUTPUT_INSTRUCTION}",
        ])

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_code_section(self, results: List[Any]) -> str:
        """Render code chunks into a budget-aware text block."""
        if not results:
            return ""

        lines: List[str] = []
        used = 0

        for i, res in enumerate(results, start=1):
            chunk = getattr(res, "chunk", res) if hasattr(res, "chunk") else res
            text  = chunk.get("text", chunk.get("content", ""))
            if not text:
                continue

            header = self._chunk_header(i, chunk)
            entry  = f"{header}\n```\n{text}\n```"
            cost   = _count_tokens(entry)

            if used + cost > self._code_budget:
                # Try a truncated version.
                remaining = self._code_budget - used
                if remaining < 50:
                    break
                entry = f"{header}\n```\n{_truncate(text, remaining - 20)}\n```"
                lines.append(entry)
                break

            lines.append(entry)
            used += cost

        return "\n\n".join(lines)

    def _build_security_section(self, results: List[Any]) -> str:
        """Render security rules into a budget-aware text block."""
        if not results:
            return ""

        lines: List[str] = []
        used  = 0
        per_rule_budget = max(200, self._security_budget // max(len(results), 1))

        for i, res in enumerate(results, start=1):
            title     = getattr(res, "title",    res.get("title", "Rule") if isinstance(res, dict) else "Rule")
            severity  = getattr(res, "severity", res.get("severity", "") if isinstance(res, dict) else "")
            rule_id   = getattr(res, "rule_id",  res.get("rule_id", "") if isinstance(res, dict) else "")
            cloud_provider = getattr(res, "cloud_provider", res.get("cloud_provider", "") if isinstance(res, dict) else "")
            resource_type  = getattr(res, "resource_type", res.get("resource_type", "") if isinstance(res, dict) else "")
            text      = getattr(res, "text",     res.get("text", "") if isinstance(res, dict) else "")

            header = f"### [{i}] {rule_id} — {title} (Severity: {severity}, Cloud: {cloud_provider}, Resource: {resource_type})"
            body   = _truncate(text, per_rule_budget)
            entry  = f"{header}\n{body}"
            cost   = _count_tokens(entry)

            if used + cost > self._security_budget:
                break

            lines.append(entry)
            used += cost

        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_header(index: int, chunk: dict) -> str:
        """Return a one-line descriptive header for a code chunk."""
        parts = [f"[{index}]"]
        if chunk.get("chunk_id"):
            parts.append(f"id={chunk['chunk_id']}")
        meta = chunk.get("metadata", {})
        if meta.get("cloud_provider"):
            parts.append(f"cloud={meta['cloud_provider']}")
        if meta.get("resource_type"):
            parts.append(f"resource={meta['resource_type']}")
        fp = chunk.get("file_path", meta.get("file_path", ""))
        if fp:
            parts.append(f"file={fp}")
        if chunk.get("file_type"):
            parts.append(f"type={chunk['file_type']}")
        score = chunk.get("score")
        if score is not None:
            parts.append(f"score={score:.3f}")
        return " | ".join(parts)

    @property
    def code_budget(self) -> int:
        """Token budget allocated to code chunks."""
        return self._code_budget

    @property
    def security_budget(self) -> int:
        """Token budget allocated to security rules."""
        return self._security_budget
