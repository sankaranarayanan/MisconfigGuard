"""
LocalLLMClient — Thin HTTP client for a locally running Ollama instance.

Ollama API reference:  https://github.com/ollama/ollama/blob/main/docs/api.md
Default endpoint:      POST http://localhost:11434/api/generate

Supported models (install with `ollama pull <model>`):
    Configure llm.model in config.yaml for the default runtime model
    mistral     — fast, good for structured output
    codellama   — optimised for code analysis
"""

import json
import logging
from typing import Iterator, Optional

import requests

from misconfigguard.config import load_llm_config

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Prompt template for security analysis
# ------------------------------------------------------------------

_SECURITY_PROMPT_TEMPLATE = """\
You are an expert security engineer specialising in Infrastructure-as-Code (IaC) \
and application configuration security.

## Retrieved Code Context
{context}

## Security Analysis Task
{query}

## Analysis Instructions
- Identify every security vulnerability or misconfiguration present in the context.
- For each finding provide:
    • Severity:       CRITICAL | HIGH | MEDIUM | LOW | INFO
    • File & snippet: Quote the exact problematic line(s).
    • Issue:          Brief description of the vulnerability (CWE / OWASP where applicable).
    • Remediation:    Concrete code-level fix or mitigation.
- If no issues are found, explicitly state: "No security issues detected in the provided context."
- Do not hallucinate issues that are not supported by the retrieved code.

## Security Analysis
"""


class LocalLLMClient:
    """
    Sends prompts to a locally running Ollama server and returns
    the model's text response.

    Supports both buffered (``generate``) and streaming
    (``stream_generate``) response modes.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Args:
            base_url:   Ollama server base URL.
            model:      Model tag to use (must be pulled with `ollama pull`).
            timeout:    HTTP request timeout in seconds.
            max_tokens: Maximum tokens in the generated response.
        """
        llm_cfg = load_llm_config()

        self.base_url = (base_url or llm_cfg.get("base_url") or "http://localhost:11434").rstrip("/")
        self.model = model or llm_cfg.get("model") or ""
        self.timeout = timeout if timeout is not None else llm_cfg.get("timeout", 600)
        self.max_tokens = max_tokens if max_tokens is not None else llm_cfg.get("max_tokens", 2048)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _generate_url(self) -> str:
        return f"{self.base_url}/api/generate"

    def _build_payload(self, prompt: str, stream: bool) -> dict:
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {"num_predict": self.max_tokens},
        }

    @staticmethod
    def _collect_stream(response: requests.Response) -> str:
        """Accumulate a streaming NDJSON response into a single string."""
        parts: list = []
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                parts.append(data.get("response", ""))
                if data.get("done"):
                    break
        return "".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list:
        """Return the names of locally available Ollama models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except requests.RequestException as exc:
            logger.error("Could not list Ollama models: %s", exc)
            return []

    def generate(self, prompt: str) -> str:
        """
        Send *prompt* to Ollama and return the complete response string.

        Raises:
            requests.Timeout:        If the request exceeds ``timeout`` seconds.
            requests.RequestException: For other HTTP / connection errors.
        """
        payload = self._build_payload(prompt, stream=True)
        logger.debug(
            "Sending prompt to %s/%s (%d chars)",
            self.model,
            self.base_url,
            len(prompt),
        )
        try:
            with requests.post(
                self._generate_url,
                json=payload,
                timeout=self.timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                return self._collect_stream(resp)
        except requests.Timeout:
            logger.error("Ollama request timed out after %ds", self.timeout)
            raise
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            raise

    def stream_generate(self, prompt: str) -> Iterator[str]:
        """
        Stream response tokens from Ollama as they are produced.

        Yields individual token strings until the model signals ``done``.
        """
        payload = self._build_payload(prompt, stream=True)
        with requests.post(
            self._generate_url,
            json=payload,
            timeout=self.timeout,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    yield data.get("response", "")
                    if data.get("done"):
                        break

    def analyze_security(
        self,
        context: str,
        query: str,
        stream: bool = False,
    ) -> str:
        """
        Run a security analysis using the retrieved code *context*.

        Args:
            context: Formatted string of retrieved code chunks.
            query:   The security question or analysis directive.
            stream:  If True, print tokens to stdout as they arrive
                     and also return the full assembled response.

        Returns:
            The model's full security analysis as a string.
        """
        prompt = _SECURITY_PROMPT_TEMPLATE.format(
            context=context, query=query
        )

        if stream:
            parts: list = []
            for token in self.stream_generate(prompt):
                print(token, end="", flush=True)
                parts.append(token)
            print()
            return "".join(parts)

        return self.generate(prompt)
