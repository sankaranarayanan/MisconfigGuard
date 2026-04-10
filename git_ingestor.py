"""
GitIngestor — Clones public and private Git repositories then yields
supported files for downstream parsing.

Authentication for private repos is handled by injecting a personal
access token into the HTTPS clone URL.
"""

import logging
import shutil
from pathlib import Path
from typing import Generator, Optional, Tuple

from git import GitCommandError, InvalidGitRepositoryError, Repo

from file_scanner import FileScanner

logger = logging.getLogger(__name__)


class GitIngestor:
    """
    Clones or updates a Git repository and yields (file_path, Repo) tuples
    for every supported file found in the working tree.
    """

    def __init__(
        self,
        clone_dir: str = "./tmp/repos",
        scanner: Optional[FileScanner] = None,
        depth: int = 1,
    ):
        """
        Args:
            clone_dir: Local directory where repos are cloned.
            scanner:   FileScanner instance (created with defaults if omitted).
            depth:     Shallow-clone depth (1 = latest snapshot only).
        """
        self.clone_dir = Path(clone_dir)
        self.clone_dir.mkdir(parents=True, exist_ok=True)
        self.scanner = scanner or FileScanner()
        self.depth = depth

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _repo_dir(self, url: str) -> Path:
        """Derive a deterministic local path from the repo URL."""
        name = url.rstrip("/").split("/")[-1].replace(".git", "")
        return self.clone_dir / name

    def _auth_url(self, url: str, token: Optional[str]) -> str:
        """Inject a PAT into an HTTPS URL for private repo access."""
        if token and url.startswith("https://"):
            return url.replace("https://", f"https://{token}@", 1)
        return url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone(
        self,
        url: str,
        token: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> Repo:
        """
        Clone *url* if not already present, or pull the latest changes.

        Args:
            url:    Repository URL (HTTPS or SSH).
            token:  Personal access token for private HTTPS repos.
            branch: Specific branch to checkout (default: remote HEAD).

        Returns:
            A GitPython ``Repo`` object pointing at the local clone.
        """
        dest = self._repo_dir(url)

        # Re-use existing clone when possible
        if dest.exists():
            logger.info("Repo already cloned at %s — pulling latest...", dest)
            try:
                repo = Repo(dest)
                repo.remotes.origin.pull()
                return repo
            except (InvalidGitRepositoryError, GitCommandError) as exc:
                logger.warning("Pull failed (%s) — re-cloning...", exc)
                shutil.rmtree(dest)

        auth = self._auth_url(url, token)
        clone_kwargs: dict = {"depth": self.depth}
        if branch:
            clone_kwargs["branch"] = branch

        logger.info("Cloning %s → %s", url, dest)
        return Repo.clone_from(auth, dest, **clone_kwargs)

    def scan_repo(
        self,
        url: str,
        token: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> Generator[Tuple[Path, Repo], None, None]:
        """
        Clone *url* and yield ``(file_path, repo)`` for every supported file.

        The generator streams results — the caller never holds all file
        paths in memory simultaneously.
        """
        repo = self.clone(url, token=token, branch=branch)
        for file_path in self.scanner.scan(str(repo.working_dir)):
            yield file_path, repo

    def get_file_metadata(self, repo: Repo, file_path: Path) -> dict:
        """
        Extract git provenance metadata for a file.

        Returns a dict compatible with the ``FileRecord.metadata`` field.
        """
        try:
            branch = repo.active_branch.name
        except TypeError:
            # Detached HEAD (common with shallow clones)
            branch = "HEAD"

        try:
            commit = repo.head.commit.hexsha[:8]
        except Exception:
            commit = "unknown"

        remote_url = ""
        try:
            remote_url = repo.remotes.origin.url
            # Strip embedded token from stored URL
            if "@" in remote_url:
                remote_url = "https://" + remote_url.split("@", 1)[1]
        except Exception:
            pass

        return {"repo": remote_url, "branch": branch, "commit": commit}
