"""Storage path helpers.

Project-scoped, local-first locations for downloaded models and streamable
chunks. Everything lives under the project's ``.tinct/`` directory
(git-ignored), so nothing sensitive is ever tracked.
"""

from __future__ import annotations

from pathlib import Path

from tinct.engine.deps import MissingDependencyError


def get_cache_dir(project_root: Path | None = None) -> Path:
    """Return the per-project cache root (``<project>/.tinct/cache``)."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    return root / ".tinct" / "cache"


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
    if not (target / "model.safetensors.index.json").is_file():
        raise ValueError(
            f"Model at {target} has no safetensors index file. "
            "tinct requires safetensors weights (pickle .bin is blocked)."
        )
    return target.resolve()


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