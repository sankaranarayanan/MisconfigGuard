"""
Chunker — Splits FileRecord content into overlapping semantic windows
(Chunk objects) suitable for embedding and vector-store ingestion.

Why overlapping windows?
    Splitting on hard boundaries risks cutting a security-relevant
    config block in half.  Overlapping adjacent chunks ensures that
    every logical unit appears fully in at least one chunk.
"""

import logging
from dataclasses import dataclass, field
from typing import Generator, Iterable, List

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    A single semantic unit of file content, ready for embedding.

    Attributes:
        text:        The raw text window.
        file_path:   Origin file path.
        file_type:   "terraform" | "yaml" | "json".
        chunk_index: 0-based position within the source file.
        metadata:    Git provenance dict from the parent FileRecord.
    """

    text: str
    file_path: str
    file_type: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for vector-store storage."""
        return {
            "text": self.text,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


class Chunker:
    """
    Splits FileRecord content into overlapping token windows.

    Tokens are approximated as whitespace-separated words — this avoids
    requiring a tokeniser dependency while staying close enough to real
    token counts for embedding models.
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        """
        Args:
            chunk_size: Maximum number of tokens (words) per chunk.
            overlap:    Number of tokens shared between adjacent chunks.

        Raises:
            ValueError: If overlap is not strictly less than chunk_size.
        """
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._step = chunk_size - overlap

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Split text into word tokens (whitespace-based)."""
        return text.split()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(self, text: str) -> Generator[str, None, None]:
        """
        Yield overlapping text windows from *text*.

        Each window contains at most ``chunk_size`` tokens and shares
        ``overlap`` tokens with its predecessor.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return

        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            yield " ".join(tokens[start:end])
            if end == len(tokens):
                break
            start += self._step

    def chunk_record(self, record: dict) -> Generator[Chunk, None, None]:
        """Yield ``Chunk`` objects derived from a single ``FileRecord``."""
        content = record.get("content", "")
        if not content.strip():
            logger.debug("Skipping empty content for %s", record.get("file_path"))
            return

        for idx, text in enumerate(self.chunk_text(content)):
            yield Chunk(
                text=text,
                file_path=record["file_path"],
                file_type=record["file_type"],
                chunk_index=idx,
                metadata=record.get("metadata", {}),
            )

    def chunk_records(
        self, records: Iterable[dict]
    ) -> Generator[Chunk, None, None]:
        """Yield ``Chunk`` objects from an iterable of ``FileRecord`` dicts."""
        for record in records:
            yield from self.chunk_record(record)
