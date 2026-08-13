"""Project initialization logic called by ``tinct init``.

Creates the full local state tree, seeds a default ``project.yaml`` when absent,
and writes a ``.gitignore`` so secret/local state under ``.tinct`` is never
tracked. Fail-closed: refuses to scaffold over a non-empty directory and rejects
unsupported model families before anything is written.
"""

from __future__ import annotations

from pathlib import Path

from tinct.core.model_gate import check_model_family
from tinct.core.project import Project
from tinct.storage.paths import TinctPaths

# Lines ensured in the project-root .gitignore (local-only state).
PROJECT_GITIGNORE = (
    ".tinct/cache/\n"
    ".tinct/runs/\n"
    ".tinct/keys/\n"
    ".tinct/*.env\n"
)


def init_tinct_project(project_root: Path, model: str,
                       project_name: str | None = None) -> Project:
    """Initialize a tinct project at ``project_root`` and return the :class:`Project`.

    Side effects (only when missing):
    - Creates the ``.tinct`` directory tree (TinctPaths.ensure_dirs).
    - Writes ``.tinct/project.yaml`` seeded with policy defaults.
    - Writes the project-root ``.gitignore``.

    Raises:
        UnsupportedModelFamily: if ``model`` is not an allowed family.
        FileExistsError: if ``project_root`` already contains files
            (fail-closed: never scaffold over existing work).
    """
    root = Path(project_root).resolve()

    # Fail-closed before any writes: model must be an allowed family.
    check_model_family(model)

    # Scaffold state dirs + a default config (delegates to Project.create for
    # fail-fast non-empty checks, state.json, and schema-valid YAML).
    project = Project.create(
        root,
        project_name=project_name or root.name,
        model=model,
    )

    _ensure_gitignore(root)
    return project


def _ensure_gitignore(root: Path) -> None:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(PROJECT_GITIGNORE, encoding="utf-8")


def paths_for(project_root: Path) -> TinctPaths:
    """Convenience: return the :class:`TinctPaths` for a project root."""
    return TinctPaths(project_root)
