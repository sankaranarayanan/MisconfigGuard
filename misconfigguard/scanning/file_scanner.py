"""
FileScanner — Recursively discovers supported IaC and config files.

Supported extensions: .tf  .yml  .yaml  .json
"""

import logging
from pathlib import Path
from typing import Generator, Optional, Set

logger = logging.getLogger(__name__)


class FileScanner:
    """
    Recursively walks a directory tree and yields paths to supported files.

    Uses a generator so that large repository trees are never fully loaded
    into memory at once.
    """

    SUPPORTED_EXTENSIONS: Set[str] = {".tf", ".yml", ".yaml", ".json"}
    DEFAULT_IGNORED_DIRS: Set[str] = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
    }

    def __init__(
        self,
        supported_extensions: Optional[Set[str]] = None,
        max_file_size_mb: float = 10.0,
        ignored_dirs: Optional[Set[str]] = None,
    ):
        """
        Args:
            supported_extensions: Override the default set of file extensions.
            max_file_size_mb:     Skip files larger than this threshold.
            ignored_dirs:         Directory names to exclude from recursive scans.
        """
        self.supported_extensions = supported_extensions or self.SUPPORTED_EXTENSIONS
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.ignored_dirs = {
            directory.lower() for directory in (ignored_dirs or self.DEFAULT_IGNORED_DIRS)
        }

    def scan(self, directory: str) -> Generator[Path, None, None]:
        """
        Yield Path objects for every supported file under *directory*.

        This is a generator — callers receive one file at a time without
        the scanner accumulating a list of all matches.
        """
        root = Path(directory)

        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not root.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        for path in root.rglob("*"):
            if any(part.lower() in self.ignored_dirs for part in path.parts):
                continue

            if not path.is_file():
                continue

            # Filter by extension
            if path.suffix.lower() not in self.supported_extensions:
                continue

            # Guard against oversized / binary-disguised files
            try:
                size = path.stat().st_size
                if size > self.max_file_size_bytes:
                    logger.warning(
                        "Skipping oversized file (%.1f MB): %s",
                        size / 1_048_576,
                        path,
                    )
                    continue
            except OSError as exc:
                logger.error("Cannot stat %s: %s", path, exc)
                continue

            yield path

    def count(self, directory: str) -> int:
        """Return the total number of supported files without storing them."""
        return sum(1 for _ in self.scan(directory))
