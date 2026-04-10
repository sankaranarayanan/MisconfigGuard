"""
IntelligentChunker — Semantic, dependency-aware code chunker for MisconfigGuard.

Splits Terraform, YAML, and JSON files into meaningful logical units rather than
naive fixed-size windows.  Each chunk carries structured dependency metadata so
the retrieval layer can expand results to include related configuration blocks.

Chunking strategies
-------------------
  "semantic"  — Split on language-level structure (resource blocks, YAML keys,
                JSON objects).  This is the default and recommended strategy.
  "fixed"     — Plain overlapping token windows (legacy behaviour, same as
                the original Chunker class).
  "hybrid"    — Semantic boundaries preferred; oversized semantic chunks are
                further split by token window as a fallback.

Sub-components
--------------
  TerraformChunker   — Regex-based HCL block extraction with ref detection.
  YAMLChunker        — PyYAML-based key / document splitting.
  JSONChunker        — json-based object / statement splitting.
  DependencyResolver — Cross-file reference registry and BFS expansion.
  IntelligentChunker — Main entry point; routes and orchestrates the above.

Chunk structure
---------------
  {
    "chunk_id":    "main_resource_aws_instance_web",
    "text":        "resource \\"aws_instance\\" \\"web\\" { ... }",
    "file_path":   "/path/to/main.tf",
    "file_type":   "terraform",
    "chunk_index": 0,
    "tokens":      320,
    "dependencies": ["aws_vpc.main", "var.instance_type"],
    "metadata": {
      "block_type":    "resource",
      "resource_type": "aws_instance",
      "block_name":    "web",
      "file_path":     "/path/to/main.tf",
      "repo":          "",
    }
  }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token counter — uses tiktoken when available, whitespace-split otherwise
# ---------------------------------------------------------------------------

try:
    import tiktoken

    _encoder = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_encoder.encode(text))

except ImportError:  # pragma: no cover
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        """Approximate token count via whitespace splitting (~4 chars/token)."""
        return max(1, len(text.split()))


# ---------------------------------------------------------------------------
# IntelligentChunk dataclass
# ---------------------------------------------------------------------------


@dataclass
class IntelligentChunk:
    """
    A semantically meaningful chunk of configuration code.

    The *text* field is intentionally named to match the legacy ``Chunk``
    interface so that existing pipeline code (``c.text``, ``c.to_dict()``)
    works without modification.
    """

    chunk_id: str
    text: str                              # chunk content
    file_path: str
    file_type: str
    chunk_index: int
    tokens: int
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for vector-store storage."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_chunk_id(base: str) -> str:
    """Return a clean, alphanumeric-underscore chunk identifier."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", base).strip("_")
    clean = re.sub(r"_+", "_", clean)
    return clean[:120]


def _token_window_split(
    text: str, max_tokens: int, overlap_tokens: int
) -> List[Tuple[str, int]]:
    """Split *text* into overlapping word-token windows."""
    words = text.split()
    if not words:
        return []
    if len(words) <= max_tokens:
        return [(text, len(words))]

    step = max(1, max_tokens - overlap_tokens)
    windows: List[Tuple[str, int]] = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        window_text = " ".join(words[start:end])
        windows.append((window_text, end - start))
        if end == len(words):
            break
        start += step
    return windows


def _maybe_split(
    text: str,
    token_count: int,
    max_tokens: int,
    overlap_tokens: int,
    strategy: str,
) -> List[Tuple[str, int]]:
    """
    Return sub-chunks of *text* according to *strategy*.

    - "fixed"  → always split into windows
    - "hybrid" → split only when *token_count* exceeds *max_tokens*
    - "semantic" → never split (caller handles size limits)
    """
    if strategy == "fixed":
        return _token_window_split(text, max_tokens, overlap_tokens)
    if strategy == "hybrid" and token_count > max_tokens:
        return _token_window_split(text, max_tokens, overlap_tokens)
    return [(text, token_count)]


def _fixed_chunks(
    content: str,
    file_path: str,
    file_type: str,
    meta: Dict[str, Any],
    max_tokens: int,
    overlap_tokens: int,
) -> List[IntelligentChunk]:
    """Fallback: fixed token-window chunking for unsupported or unparseable files."""
    file_stem = Path(file_path).stem
    repo = meta.get("repo", "")
    windows = _token_window_split(content, max_tokens, overlap_tokens)
    if not windows:
        return []
    return [
        IntelligentChunk(
            chunk_id=_make_chunk_id(f"{file_stem}_{file_type}_fixed_{i}"),
            text=text,
            file_path=file_path,
            file_type=file_type,
            chunk_index=i,
            tokens=tokens,
            dependencies=[],
            metadata={
                "block_type": "fixed_window",
                "file_path": file_path,
                "repo": repo,
            },
        )
        for i, (text, tokens) in enumerate(windows)
    ]


# ---------------------------------------------------------------------------
# TerraformChunker
# ---------------------------------------------------------------------------

# Matches the opening header of any top-level Terraform block.
_TF_BLOCK_RE = re.compile(
    r"^(resource|module|variable|output|data|provider|locals|terraform)\b"
    r'(?:\s+"([^"]*)")?(?:\s+"([^"]*)")?\s*\{',
    re.MULTILINE,
)

# Reference pattern matchers for dependency extraction.
_TF_VAR_REF    = re.compile(r"\bvar\.([a-zA-Z_]\w*)\b")
_TF_LOCAL_REF  = re.compile(r"\blocal\.([a-zA-Z_]\w*)\b")
_TF_MODULE_REF = re.compile(r"\bmodule\.([a-zA-Z_]\w*)\b")
_TF_DATA_REF   = re.compile(r"\bdata\.([a-zA-Z_]\w+)\.([a-zA-Z_]\w+)\b")
# Resource references look like  aws_vpc.main  — filter out built-in keywords.
_TF_RES_REF    = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\.([a-zA-Z_]\w*)\b")
_TF_EXCLUDED   = frozenset({
    "var", "local", "module", "data", "path", "terraform", "self",
    "each", "count", "toset", "tolist", "tomap", "merge", "concat",
    "lookup", "length", "format", "join", "split", "trimspace",
})


def _find_block_end(text: str, brace_start: int) -> int:
    """Return the index of the matching closing ``}`` for an open brace."""
    depth = 0
    in_string = False
    i = brace_start
    while i < len(text):
        ch = text[i]
        # Toggle string mode on unescaped double-quote.
        if ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return len(text) - 1  # malformed HCL fallback


def _extract_tf_dependencies(content: str) -> List[str]:
    """Extract all dependency references from a Terraform block body."""
    deps: set = set()
    for m in _TF_VAR_REF.finditer(content):
        deps.add(f"var.{m.group(1)}")
    for m in _TF_LOCAL_REF.finditer(content):
        deps.add(f"local.{m.group(1)}")
    for m in _TF_MODULE_REF.finditer(content):
        deps.add(f"module.{m.group(1)}")
    for m in _TF_DATA_REF.finditer(content):
        deps.add(f"data.{m.group(1)}.{m.group(2)}")
    for m in _TF_RES_REF.finditer(content):
        if m.group(1) not in _TF_EXCLUDED:
            deps.add(f"{m.group(1)}.{m.group(2)}")
    return sorted(deps)


class TerraformChunker:
    """Splits Terraform HCL files into one chunk per top-level block."""

    def chunk(
        self,
        record: Dict[str, Any],
        max_tokens: int,
        overlap_tokens: int,
        strategy: str,
    ) -> List[IntelligentChunk]:
        content: str = record["content"]
        file_path: str = record["file_path"]
        repo: str = record.get("metadata", {}).get("repo", "")
        file_stem = Path(file_path).stem

        chunks: List[IntelligentChunk] = []

        for match in _TF_BLOCK_RE.finditer(content):
            block_type = match.group(1)
            # label1 = resource-type OR module/variable/output name
            label1 = match.group(2) or ""
            # label2 = resource name (only populated for resource / data blocks)
            label2 = match.group(3) or ""

            # Locate the opening brace and extract the full block.
            brace_pos = content.index("{", match.start())
            block_end = _find_block_end(content, brace_pos)
            block_text = content[match.start(): block_end + 1]

            if block_type in ("resource", "data"):
                resource_type = label1
                block_name    = label2
            else:
                resource_type = ""
                block_name    = label1

            # Build a deterministic, human-readable chunk_id.
            id_parts = [file_stem, block_type]
            if resource_type:
                id_parts.append(resource_type)
            if block_name:
                id_parts.append(block_name)
            base_id = _make_chunk_id("_".join(id_parts))

            token_count = _count_tokens(block_text)
            deps = _extract_tf_dependencies(block_text)

            sub_chunks = _maybe_split(
                block_text, token_count, max_tokens, overlap_tokens, strategy
            )

            for i, (sub_text, sub_tokens) in enumerate(sub_chunks):
                chunk_id = base_id if len(sub_chunks) == 1 else f"{base_id}_{i}"
                chunks.append(
                    IntelligentChunk(
                        chunk_id=chunk_id,
                        text=sub_text,
                        file_path=file_path,
                        file_type="terraform",
                        chunk_index=len(chunks),
                        tokens=sub_tokens,
                        dependencies=deps,
                        metadata={
                            "block_type":    block_type,
                            "resource_type": resource_type,
                            "block_name":    block_name,
                            "file_path":     file_path,
                            "repo":          repo,
                        },
                    )
                )

        if not chunks:
            logger.debug(
                "No Terraform blocks found in %s; falling back to fixed chunking",
                file_path,
            )
            chunks = _fixed_chunks(
                content, file_path, "terraform",
                record.get("metadata", {}), max_tokens, overlap_tokens,
            )

        return chunks


# ---------------------------------------------------------------------------
# YAMLChunker
# ---------------------------------------------------------------------------


def _yaml_to_text(data: Any, key: Optional[str] = None) -> str:
    """Re-serialise a parsed YAML value back to a YAML string."""
    payload = {key: data} if key is not None else data
    return yaml.dump(payload, default_flow_style=False, allow_unicode=True)


# Detect reference-like expressions inside YAML string values.
_YAML_REF_RE = re.compile(r"\$\{([^}]+)\}|\bvar\.(\w+)\b|\bRef:\s*(\S+)")


def _extract_yaml_deps(data: Any) -> List[str]:
    """Walk a parsed YAML structure and collect reference-like strings."""
    refs: set = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, str):
            for m in _YAML_REF_RE.finditer(obj):
                ref = m.group(1) or m.group(2) or m.group(3)
                if ref:
                    refs.add(ref.strip())

    walk(data)
    return sorted(refs)


class YAMLChunker:
    """Splits YAML files into semantic chunks based on top-level keys or documents."""

    def chunk(
        self,
        record: Dict[str, Any],
        max_tokens: int,
        overlap_tokens: int,
        strategy: str,
    ) -> List[IntelligentChunk]:
        content: str = record["content"]
        file_path: str = record["file_path"]
        repo: str = record.get("metadata", {}).get("repo", "")
        file_stem = Path(file_path).stem

        chunks: List[IntelligentChunk] = []

        try:
            documents = list(yaml.safe_load_all(content))
        except yaml.YAMLError as exc:
            logger.warning("YAML parse error in %s: %s", file_path, exc)
            return _fixed_chunks(
                content, file_path, "yaml",
                record.get("metadata", {}), max_tokens, overlap_tokens,
            )

        # Split the raw source into per-document text for context preservation.
        raw_docs = self._split_raw_documents(content)
        # Zip defensively — pad raw_docs if fewer separators than documents.
        while len(raw_docs) < len(documents):
            raw_docs.append("")

        for doc_idx, (doc_data, doc_text) in enumerate(zip(documents, raw_docs)):
            if doc_data is None:
                continue

            # Kubernetes / CRD manifests: treat the whole document as one chunk.
            if isinstance(doc_data, dict) and "kind" in doc_data:
                kind = doc_data.get("kind", "unknown")
                name = (doc_data.get("metadata") or {}).get("name", str(doc_idx))
                base_id = _make_chunk_id(f"{file_stem}_yaml_{kind}_{name}")
                chunk_text = doc_text or _yaml_to_text(doc_data)
                token_count = _count_tokens(chunk_text)

                sub_chunks = _maybe_split(
                    chunk_text, token_count, max_tokens, overlap_tokens, strategy
                )
                for i, (sub_text, sub_tokens) in enumerate(sub_chunks):
                    chunk_id = base_id if len(sub_chunks) == 1 else f"{base_id}_{i}"
                    chunks.append(
                        IntelligentChunk(
                            chunk_id=chunk_id,
                            text=sub_text,
                            file_path=file_path,
                            file_type="yaml",
                            chunk_index=len(chunks),
                            tokens=sub_tokens,
                            dependencies=_extract_yaml_deps(doc_data),
                            metadata={
                                "block_type":    "k8s_document",
                                "block_name":    name,
                                "resource_type": kind,
                                "file_path":     file_path,
                                "repo":          repo,
                            },
                        )
                    )
                continue

            # Generic mapping: one chunk per top-level key.
            if isinstance(doc_data, dict):
                for key, value in doc_data.items():
                    key_text = _yaml_to_text(value, str(key))
                    base_id = _make_chunk_id(f"{file_stem}_yaml_{doc_idx}_{key}")
                    token_count = _count_tokens(key_text)

                    sub_chunks = _maybe_split(
                        key_text, token_count, max_tokens, overlap_tokens, strategy
                    )
                    for i, (sub_text, sub_tokens) in enumerate(sub_chunks):
                        chunk_id = base_id if len(sub_chunks) == 1 else f"{base_id}_{i}"
                        chunks.append(
                            IntelligentChunk(
                                chunk_id=chunk_id,
                                text=sub_text,
                                file_path=file_path,
                                file_type="yaml",
                                chunk_index=len(chunks),
                                tokens=sub_tokens,
                                dependencies=_extract_yaml_deps(value),
                                metadata={
                                    "block_type": "yaml_key",
                                    "block_name": str(key),
                                    "file_path":  file_path,
                                    "repo":       repo,
                                },
                            )
                        )

            # Root-level list: one chunk per element.
            elif isinstance(doc_data, list):
                for i, item in enumerate(doc_data):
                    item_text = _yaml_to_text(item)
                    base_id = _make_chunk_id(f"{file_stem}_yaml_{doc_idx}_item_{i}")
                    token_count = _count_tokens(item_text)
                    chunks.append(
                        IntelligentChunk(
                            chunk_id=base_id,
                            text=item_text,
                            file_path=file_path,
                            file_type="yaml",
                            chunk_index=len(chunks),
                            tokens=token_count,
                            dependencies=_extract_yaml_deps(item),
                            metadata={
                                "block_type": "yaml_list_item",
                                "block_name": str(i),
                                "file_path":  file_path,
                                "repo":       repo,
                            },
                        )
                    )

            else:
                # Scalar at root — unlikely but handled gracefully.
                scalar_text = str(doc_data)
                base_id = _make_chunk_id(f"{file_stem}_yaml_{doc_idx}_scalar")
                chunks.append(
                    IntelligentChunk(
                        chunk_id=base_id,
                        text=scalar_text,
                        file_path=file_path,
                        file_type="yaml",
                        chunk_index=len(chunks),
                        tokens=_count_tokens(scalar_text),
                        dependencies=[],
                        metadata={
                            "block_type": "yaml_scalar",
                            "file_path":  file_path,
                            "repo":       repo,
                        },
                    )
                )

        if not chunks:
            chunks = _fixed_chunks(
                content, file_path, "yaml",
                record.get("metadata", {}), max_tokens, overlap_tokens,
            )

        return chunks

    @staticmethod
    def _split_raw_documents(content: str) -> List[str]:
        """Split multi-document YAML on ``---`` separators."""
        parts = re.split(r"^---\s*$", content, flags=re.MULTILINE)
        return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# JSONChunker
# ---------------------------------------------------------------------------

# CloudFormation / CDK reference patterns.
_JSON_REF_RE = re.compile(r"\$\{([^}]+)\}|\"Ref\"\s*:\s*\"([^\"]+)\"|arn:[^\s\"]+")


def _extract_json_deps(data: Any) -> List[str]:
    """Walk a parsed JSON structure and collect reference-like strings."""
    refs: set = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            # CloudFormation Ref / Fn::GetAtt
            if "Ref" in obj and isinstance(obj["Ref"], str):
                refs.add(obj["Ref"])
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, str):
            for m in _JSON_REF_RE.finditer(obj):
                ref = m.group(1) or m.group(2) or m.group(0)
                if ref:
                    refs.add(ref.strip())

    walk(data)
    return sorted(refs)


class JSONChunker:
    """Splits JSON files into semantic chunks by top-level key or IAM statement."""

    def chunk(
        self,
        record: Dict[str, Any],
        max_tokens: int,
        overlap_tokens: int,
        strategy: str,
    ) -> List[IntelligentChunk]:
        content: str = record["content"]
        file_path: str = record["file_path"]
        repo: str = record.get("metadata", {}).get("repo", "")
        file_stem = Path(file_path).stem

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error in %s: %s", file_path, exc)
            return _fixed_chunks(
                content, file_path, "json",
                record.get("metadata", {}), max_tokens, overlap_tokens,
            )

        chunks: List[IntelligentChunk] = []

        # IAM Policy document: split into per-Statement chunks.
        if (
            isinstance(data, dict)
            and "Statement" in data
            and isinstance(data["Statement"], list)
        ):
            # Header chunk (everything except Statement array).
            header = {k: v for k, v in data.items() if k != "Statement"}
            if header:
                header_text = json.dumps(header, indent=2)
                chunks.append(
                    IntelligentChunk(
                        chunk_id=_make_chunk_id(f"{file_stem}_json_iam_header"),
                        text=header_text,
                        file_path=file_path,
                        file_type="json",
                        chunk_index=0,
                        tokens=_count_tokens(header_text),
                        dependencies=[],
                        metadata={
                            "block_type": "iam_policy_header",
                            "file_path":  file_path,
                            "repo":       repo,
                        },
                    )
                )
            for i, stmt in enumerate(data["Statement"]):
                stmt_text = json.dumps(stmt, indent=2)
                sid = stmt.get("Sid", str(i))
                chunk_id = _make_chunk_id(f"{file_stem}_json_statement_{sid}")
                chunks.append(
                    IntelligentChunk(
                        chunk_id=chunk_id,
                        text=stmt_text,
                        file_path=file_path,
                        file_type="json",
                        chunk_index=len(chunks),
                        tokens=_count_tokens(stmt_text),
                        dependencies=_extract_json_deps(stmt),
                        metadata={
                            "block_type": "iam_statement",
                            "block_name": str(sid),
                            "file_path":  file_path,
                            "repo":       repo,
                        },
                    )
                )
            return chunks

        # Generic object: one chunk per top-level key.
        if isinstance(data, dict):
            for key, value in data.items():
                key_text = json.dumps({key: value}, indent=2)
                base_id = _make_chunk_id(f"{file_stem}_json_{key}")
                token_count = _count_tokens(key_text)

                sub_chunks = _maybe_split(
                    key_text, token_count, max_tokens, overlap_tokens, strategy
                )
                for i, (sub_text, sub_tokens) in enumerate(sub_chunks):
                    chunk_id = base_id if len(sub_chunks) == 1 else f"{base_id}_{i}"
                    chunks.append(
                        IntelligentChunk(
                            chunk_id=chunk_id,
                            text=sub_text,
                            file_path=file_path,
                            file_type="json",
                            chunk_index=len(chunks),
                            tokens=sub_tokens,
                            dependencies=_extract_json_deps(value),
                            metadata={
                                "block_type": "json_key",
                                "block_name": str(key),
                                "file_path":  file_path,
                                "repo":       repo,
                            },
                        )
                    )

        # Root array: one chunk per element.
        elif isinstance(data, list):
            for i, item in enumerate(data):
                item_text = json.dumps(item, indent=2)
                base_id = _make_chunk_id(f"{file_stem}_json_item_{i}")
                chunks.append(
                    IntelligentChunk(
                        chunk_id=base_id,
                        text=item_text,
                        file_path=file_path,
                        file_type="json",
                        chunk_index=len(chunks),
                        tokens=_count_tokens(item_text),
                        dependencies=_extract_json_deps(item),
                        metadata={
                            "block_type": "json_array_item",
                            "block_name": str(i),
                            "file_path":  file_path,
                            "repo":       repo,
                        },
                    )
                )

        if not chunks:
            chunks = _fixed_chunks(
                content, file_path, "json",
                record.get("metadata", {}), max_tokens, overlap_tokens,
            )

        return chunks


# ---------------------------------------------------------------------------
# DependencyResolver
# ---------------------------------------------------------------------------


class DependencyResolver:
    """
    Maintains a registry of all chunks and resolves symbolic dependency
    references (e.g. ``"var.instance_type"``) to their owning chunk_ids.

    Usage
    -----
    1. After chunking each file, call ``register(chunks)``.
    2. After all files are chunked, call ``resolve(all_chunks)`` to replace
       symbolic references with chunk_ids where mappings are known.
    3. During retrieval, call ``expand_dependencies(chunk_ids)`` to pull in
       related chunks transitively.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, IntelligentChunk] = {}
        # Maps symbolic ref (e.g. "var.foo") → chunk_id
        self._ref_map: Dict[str, str] = {}

    def register(self, chunks: List[IntelligentChunk]) -> None:
        """Index *chunks* and build the symbolic-reference lookup table."""
        for chunk in chunks:
            self._registry[chunk.chunk_id] = chunk
            block_type    = chunk.metadata.get("block_type", "")
            block_name    = chunk.metadata.get("block_name", "")
            resource_type = chunk.metadata.get("resource_type", "")

            if block_type == "variable" and block_name:
                self._ref_map[f"var.{block_name}"] = chunk.chunk_id
            elif block_type == "output" and block_name:
                self._ref_map[f"output.{block_name}"] = chunk.chunk_id
            elif block_type == "module" and block_name:
                self._ref_map[f"module.{block_name}"] = chunk.chunk_id
            elif block_type == "locals":
                self._ref_map["locals"] = chunk.chunk_id
            elif block_type == "resource" and resource_type and block_name:
                self._ref_map[f"{resource_type}.{block_name}"] = chunk.chunk_id
            elif block_type == "data" and resource_type and block_name:
                self._ref_map[f"data.{resource_type}.{block_name}"] = chunk.chunk_id

    def resolve(self, chunks: List[IntelligentChunk]) -> List[IntelligentChunk]:
        """
        Return a new list of chunks where each dependency string is replaced
        by the corresponding chunk_id if a mapping exists.
        """
        result = []
        for chunk in chunks:
            resolved_deps = [
                self._ref_map.get(dep, dep) for dep in chunk.dependencies
            ]
            result.append(
                IntelligentChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    file_path=chunk.file_path,
                    file_type=chunk.file_type,
                    chunk_index=chunk.chunk_index,
                    tokens=chunk.tokens,
                    dependencies=resolved_deps,
                    metadata=chunk.metadata,
                )
            )
        return result

    def get_chunk(self, chunk_id: str) -> Optional[IntelligentChunk]:
        """Return the chunk registered under *chunk_id*, or ``None``."""
        return self._registry.get(chunk_id)

    def expand_dependencies(
        self,
        chunk_ids: List[str],
        max_depth: int = 2,
    ) -> List[IntelligentChunk]:
        """
        BFS expansion: return the requested chunks *plus* their transitive
        dependencies up to *max_depth* hops.

        This is called during retrieval to enrich top-k results with
        related configuration blocks.
        """
        seen: set = set()
        queue = list(chunk_ids)
        result: List[IntelligentChunk] = []
        depth = 0

        while queue and depth < max_depth:
            next_queue: List[str] = []
            for cid in queue:
                if cid in seen:
                    continue
                seen.add(cid)
                chunk = self._registry.get(cid)
                if chunk:
                    result.append(chunk)
                    for dep in chunk.dependencies:
                        if dep not in seen:
                            next_queue.append(dep)
            queue = next_queue
            depth += 1

        return result

    @property
    def dependency_graph(self) -> Dict[str, List[str]]:
        """Return the full chunk → dependency adjacency list."""
        return {cid: c.dependencies for cid, c in self._registry.items()}

    @property
    def total_chunks(self) -> int:
        return len(self._registry)


# ---------------------------------------------------------------------------
# IntelligentChunker — main entry point
# ---------------------------------------------------------------------------


class IntelligentChunker:
    """
    Semantic, dependency-aware chunker for Terraform, YAML, and JSON files.

    Drop-in replacement for the legacy ``Chunker`` class: exposes the same
    ``chunk_record`` / ``chunk_records`` interface so ``RAGPipeline`` requires
    only minimal changes.

    Parameters
    ----------
    max_tokens_per_chunk : int
        Soft ceiling on tokens per chunk.  Semantic chunks that exceed this
        are further split in "hybrid" mode.  Default: 500.
    overlap_tokens : int
        Token overlap between adjacent windows when splitting large chunks.
        Default: 50.
    chunking_strategy : str
        One of "semantic" (default), "fixed", or "hybrid".
    resolve_dependencies : bool
        When True (default), register chunks with the ``DependencyResolver``
        so that cross-file references can be resolved at query time.

    Example
    -------
    >>> from intelligent_chunker import IntelligentChunker
    >>> chunker = IntelligentChunker(max_tokens_per_chunk=500, chunking_strategy="hybrid")
    >>> record = {
    ...     "file_path": "main.tf",
    ...     "file_type": "terraform",
    ...     "content": 'resource "aws_instance" "web" { ami = "ami-123" }',
    ...     "metadata": {"repo": "", "branch": "", "commit": ""},
    ... }
    >>> chunks = chunker.chunk_record(record)
    >>> print(chunks[0].chunk_id, chunks[0].dependencies)
    main_resource_aws_instance_web []
    """

    VALID_STRATEGIES = {"semantic", "fixed", "hybrid"}

    def __init__(
        self,
        max_tokens_per_chunk: int = 500,
        overlap_tokens: int = 50,
        chunking_strategy: str = "semantic",
        resolve_dependencies: bool = True,
    ) -> None:
        if chunking_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"chunking_strategy must be one of {self.VALID_STRATEGIES}, "
                f"got {chunking_strategy!r}"
            )
        if overlap_tokens >= max_tokens_per_chunk:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be less than "
                f"max_tokens_per_chunk ({max_tokens_per_chunk})"
            )

        self.max_tokens      = max_tokens_per_chunk
        self.overlap         = overlap_tokens
        self.strategy        = chunking_strategy
        self.resolve_deps    = resolve_dependencies

        self._tf_chunker     = TerraformChunker()
        self._yaml_chunker   = YAMLChunker()
        self._json_chunker   = JSONChunker()
        self.resolver        = DependencyResolver()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_record(self, record: Dict[str, Any]) -> List[IntelligentChunk]:
        """
        Chunk a single FileRecord into ``IntelligentChunk`` objects.

        Routing:
          terraform → TerraformChunker
          yaml      → YAMLChunker
          json      → JSONChunker
          other     → fixed-window fallback
        """
        file_type = record.get("file_type", "")
        try:
            if file_type == "terraform":
                chunks = self._tf_chunker.chunk(
                    record, self.max_tokens, self.overlap, self.strategy
                )
            elif file_type == "yaml":
                chunks = self._yaml_chunker.chunk(
                    record, self.max_tokens, self.overlap, self.strategy
                )
            elif file_type == "json":
                chunks = self._json_chunker.chunk(
                    record, self.max_tokens, self.overlap, self.strategy
                )
            else:
                chunks = _fixed_chunks(
                    record.get("content", ""),
                    record.get("file_path", "unknown"),
                    file_type or "unknown",
                    record.get("metadata", {}),
                    self.max_tokens,
                    self.overlap,
                )
        except Exception as exc:
            logger.warning(
                "Chunking failed for %s (%s): %s — falling back to fixed windows",
                record.get("file_path"),
                file_type,
                exc,
            )
            chunks = _fixed_chunks(
                record.get("content", ""),
                record.get("file_path", "unknown"),
                file_type or "unknown",
                record.get("metadata", {}),
                self.max_tokens,
                self.overlap,
            )

        if self.resolve_deps:
            self.resolver.register(chunks)

        return chunks

    def chunk_records(
        self, records: Iterable[Dict[str, Any]]
    ) -> Generator[IntelligentChunk, None, None]:
        """
        Chunk an iterable of FileRecords; resolve cross-file dependencies
        after all records have been processed.

        Yields ``IntelligentChunk`` objects one at a time (generator).
        """
        all_chunks: List[IntelligentChunk] = []
        records_list = list(records)

        for record in records_list:
            all_chunks.extend(self.chunk_record(record))

        # Resolve symbolic refs to chunk_ids now that all files are registered.
        if self.resolve_deps and len(records_list) > 1:
            all_chunks = self.resolver.resolve(all_chunks)

        yield from all_chunks
