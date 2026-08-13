"""Storage path helpers — the single source of truth for the local state tree.

Layout (everything under ``<project>/.tinct``, git-ignored)::

    project/
    ├── data/                    # user datasets (unmanaged)
    └── .tinct/
        ├── project.yaml         # project config (see core/config.py)
        ├── keys/                # Ed25519 ship-signing keys (0700 on POSIX)
        ├── evidence/            # signed ship evidence reports
        ├── runs/<name>/         # isolated, auditable per-run artifacts
        └── cache/
            ├── models/          # downloaded/snapshotted base models
            ├── datasets/        # dataset caches
            └── chunks/          # streamable model chunks (ModelChunker)
"""

from __future__ import annotations

import os
from pathlib import Path

from tinct.engine.deps import MissingDependencyError

# Filenames used across the state tree.
PROJECT_CONFIG_NAME = "project.yaml"


class TinctPaths:
    """Manages all local file paths for the tinct project."""

    def __init__(self, project_root: Path | str | None = None) -> None:
        # If no root provided, look for .tinct in current or parent dirs,
        # defaulting to cwd.
        self.project_root = Path(project_root or self._find_project_root())
        self.tinct_dir = self.project_root / ".tinct"

    # -- discovery ----------------------------------------------------------

    @classmethod
    def discover(cls) -> "TinctPaths":
        """Build paths from the nearest enclosing project (cwd walk-up)."""
        return cls()

    def _find_project_root(self) -> Path:
        current = Path.cwd()
        while current != current.parent:
            if (current / ".tinct").is_dir():
                return current
            current = current.parent
        return Path.cwd()

    # -- path properties ----------------------------------------------------

    @property
    def cache_dir(self) -> Path:
        return self.tinct_dir / "cache"

    @property
    def model_cache(self) -> Path:
        return self.cache_dir / "models"

    @property
    def dataset_cache(self) -> Path:
        return self.cache_dir / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.tinct_dir / "runs"

    @property
    def keys_dir(self) -> Path:
        return self.tinct_dir / "keys"

    @property
    def evidence_dir(self) -> Path:
        return self.tinct_dir / "evidence"

    @property
    def project_config(self) -> Path:
        return self.tinct_dir / PROJECT_CONFIG_NAME

    # -- lifecycle ----------------------------------------------------------

    def ensure_dirs(self) -> None:
        """Create the full .tinct directory tree."""
        for d in (
            self.cache_dir,
            self.model_cache,
            self.dataset_cache,
            self.runs_dir,
            self.keys_dir,
            self.evidence_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # Secure permissions on the keys directory (Unix only).
        if os.name != "nt" and self.keys_dir.exists():
            os.chmod(self.keys_dir, 0o700)


# Convenience shims (used by the train/CLI flows before a Project exists).


def get_cache_dir(project_root: Path | str | None = None) -> Path:
    """Return the per-project cache root (``<project>/.tinct/cache``)."""
    return TinctPaths(project_root).cache_dir


def resolve_model(model_id: str, cache_dir: Path | None = None) -> Path:
    """Return a local directory for ``model_id``, downloading it into the local
    cache first when it is a Hugging Face hub id.

    Raises:
        ValueError: If the resolved directory lacks a safetensors index
            (fail-closed: tinct only runs on safetensors; pickled ``.bin``
            checkpoints are blocked by the chunker regardless).
        MissingDependencyError: When ``model_id`` is a hub id but the HF
            client is not installed (``pip install tinct[train]``).
    """
    local = Path(model_id)
    if local.is_dir():
        target = local
    else:
        target = _snapshot_hub(hub_id=model_id, cache_dir=cache_dir)
    if not (_has_safetensors(target)):
        raise ValueError(
            f"Model at {target} has no safetensors weights (need a "
            "model.safetensors file or a model.safetensors.index.json). "
            "tinct requires safetensors (pickle .bin is blocked)."
        )
    return target.resolve()


def _has_safetensors(model_dir: Path) -> bool:
    """True if the dir holds safetensors weights (single file or sharded)."""
    if model_dir.is_dir():
        if (model_dir / "model.safetensors.index.json").is_file():
            return True
        return any(model_dir.glob("*.safetensors"))
    return False


def _snapshot_hub(hub_id: str, cache_dir: Path | None) -> Path:
    """Download a HF hub repo into the cache and return its local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on env
        raise MissingDependencyError(
            "Downloading a model from the Hub requires huggingface_hub, which "
            "is not installed. Install it with:\n\n    pip install tinct[train]\n"
        ) from exc
    cache_root = cache_dir or get_cache_dir()
    return Path(
        snapshot_download(repo_id=hub_id, cache_dir=str(cache_root / "hub"))
    )