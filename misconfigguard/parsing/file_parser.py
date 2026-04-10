"""
FileParser — Reads supported IaC files, validates their content, and
normalises them into FileRecord dicts for downstream processing.

FileRecord schema:
    {
        "file_path": str,          # absolute or relative path
        "file_type": str,          # "terraform" | "yaml" | "json"
        "content":   str,          # raw file text
        "metadata":  {
            "repo":   str,         # remote URL (empty for local files)
            "branch": str,
            "commit": str,
        }
    }
"""

import json
import logging
from pathlib import Path
from typing import Generator, Optional

import yaml

from file_scanner import FileScanner
from git_ingestor import GitIngestor

logger = logging.getLogger(__name__)

# Type alias for clarity
FileRecord = dict


class FileParser:
    """
    Parses and normalises supported file types into ``FileRecord`` dicts.

    Files are read in configurable byte-chunks to avoid loading entire
    large files into memory in one shot.
    """

    # Map file extensions → canonical type names
    EXTENSION_TYPE_MAP: dict = {
        ".tf": "terraform",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".json": "json",
    }

    def __init__(self, read_chunk_bytes: int = 8192):
        """
        Args:
            read_chunk_bytes: Bytes to read per iteration when streaming files.
        """
        self.read_chunk_bytes = read_chunk_bytes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stream_read(self, path: Path) -> str:
        """Read a file in chunks and return the complete text."""
        parts: list[str] = []  # chunk streams
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            while True:
                chunk = fh.read(self.read_chunk_bytes)
                if not chunk:
                    break
                parts.append(chunk)
        return "".join(parts)

    def _is_valid(self, path: Path, content: str, file_type: str) -> bool:
        """
        Light validation gate:
          - Skip empty files.
          - Parse YAML/JSON to catch corrupted syntax early.
          - Terraform files are kept as-is (no standard parser required).
        """
        if not content.strip():
            logger.warning("Skipping empty file: %s", path)
            return False

        if file_type == "yaml":
            try:
                list(yaml.safe_load_all(content))
            except yaml.YAMLError as exc:
                logger.error("YAML parse error in %s: %s", path, exc)
                return False

        elif file_type == "json":
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                logger.error("JSON parse error in %s: %s", path, exc)
                return False

        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file(
        self,
        file_path: Path,
        metadata: Optional[dict] = None,
    ) -> Optional[FileRecord]:
        """
        Parse a single file into a ``FileRecord``.

        Returns ``None`` if the file is unsupported, empty, or corrupt.
        """
        ext = file_path.suffix.lower()
        file_type = self.EXTENSION_TYPE_MAP.get(ext)
        if file_type is None:
            logger.debug("Unsupported extension '%s': %s", ext, file_path)
            return None

        try:
            content = self._stream_read(file_path)
        except (OSError, IOError) as exc:
            logger.error("Cannot read %s: %s", file_path, exc)
            return None

        if not self._is_valid(file_path, content, file_type):
            return None

        return {
            "file_path": str(file_path),
            "file_type": file_type,
            "content": content,
            "metadata": metadata or {"repo": "", "branch": "", "commit": ""},
        }

    def parse_directory(
        self,
        directory: str,
        scanner: Optional[FileScanner] = None,
    ) -> Generator[FileRecord, None, None]:
        """
        Yield ``FileRecord`` dicts for all supported files under *directory*.

        Results are streamed — files are parsed one at a time.
        """
        scanner = scanner or FileScanner()
        for file_path in scanner.scan(directory):
            record = self.parse_file(file_path)
            if record is not None:
                yield record

    def parse_repository(
        self,
        ingestor: GitIngestor,
        url: str,
        token: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> Generator[FileRecord, None, None]:
        """
        Clone *url* via *ingestor* and yield ``FileRecord`` dicts with
        full git provenance metadata attached.
        """
        for file_path, repo in ingestor.scan_repo(url, token=token, branch=branch):
            metadata = ingestor.get_file_metadata(repo, file_path)
            record = self.parse_file(file_path, metadata=metadata)
            if record is not None:
                yield record
