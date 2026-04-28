"""Repository scanner for ingesting source files."""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path

from codebase_ai.config import ScannerConfig
from codebase_ai.models import ScanFilters, SourceFile

logger = logging.getLogger(__name__)


class CodebaseScanner:
    """Recursively scans a repository and returns supported source files."""

    def __init__(self, config: ScannerConfig | None = None) -> None:
        self.config = config or ScannerConfig()

    def scan(
        self,
        repo_path: str | Path,
        filters: ScanFilters | None = None,
    ) -> list[SourceFile]:
        """Scan a repository path and return supported source files."""

        root = Path(repo_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Repository path does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {root}")

        active_filters = filters or ScanFilters()
        max_files = active_filters.max_files or self.config.max_files
        logger.info("Scanning repository: %s", root)
        discovered: list[SourceFile] = []

        for file_path in self._iter_source_files(root, active_filters):
            source_file = self._build_source_file(root=root, file_path=file_path)
            if source_file is not None:
                discovered.append(source_file)
            if max_files and len(discovered) >= max_files:
                logger.info("Reached file limit (%s); stopping early", max_files)
                break

        discovered.sort(key=lambda item: item.path)
        logger.info("Discovered %s supported files", len(discovered))
        return discovered

    def _iter_source_files(self, root: Path, filters: ScanFilters):
        for current_root, dirnames, filenames in os.walk(
            root,
            followlinks=self.config.follow_symlinks,
        ):
            dirnames[:] = sorted(
                directory
                for directory in dirnames
                if directory not in self.config.ignored_directories
            )

            current_dir = Path(current_root)
            for filename in sorted(filenames):
                if filename in self.config.ignored_file_names:
                    continue

                file_path = current_dir / filename
                extension = file_path.suffix.lower()
                if extension not in self.config.supported_extensions:
                    continue
                relative_path = str(file_path.relative_to(root))
                language = self.config.supported_extensions[extension]
                if not self._matches_filters(relative_path, language, filters):
                    continue

                yield file_path

    def _build_source_file(self, root: Path, file_path: Path) -> SourceFile | None:
        try:
            file_size = file_path.stat().st_size
            if file_size > self.config.max_file_size_bytes:
                logger.warning(
                    "Skipping large file (%s bytes): %s",
                    file_size,
                    file_path,
                )
                return None

            content = file_path.read_text(
                encoding=self.config.default_encoding,
                errors="ignore",
            )
        except UnicodeDecodeError:
            logger.warning("Skipping file with unsupported encoding: %s", file_path)
            return None
        except OSError as exc:
            logger.warning("Skipping unreadable file %s: %s", file_path, exc)
            return None

        relative_path = str(file_path.relative_to(root))
        language = self.config.supported_extensions[file_path.suffix.lower()]
        return SourceFile(
            path=relative_path,
            language=language,
            content=content,
            size_bytes=file_size,
            line_count=content.count("\n") + (1 if content else 0),
        )

    def _matches_filters(
        self,
        relative_path: str,
        language: str,
        filters: ScanFilters,
    ) -> bool:
        normalized_path = relative_path.replace(os.sep, "/")

        if any(
            fnmatch.fnmatch(normalized_path, pattern)
            for pattern in self.config.ignored_path_patterns
        ):
            return False

        if filters.languages and language not in filters.languages:
            return False

        if filters.include_globs and not any(
            fnmatch.fnmatch(normalized_path, pattern)
            for pattern in filters.include_globs
        ):
            return False

        if any(
            fnmatch.fnmatch(normalized_path, pattern)
            for pattern in filters.exclude_globs
        ):
            return False

        return True
