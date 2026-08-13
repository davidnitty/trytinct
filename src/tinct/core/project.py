"""Project lifecycle and on-disk state layout.

The canonical state tree lives under ``.tinct/`` and is managed by
:class:`tinct.storage.paths.TinctPaths` — see that module for the layout::

    project/
    ├── data/                    # user datasets (unmanaged)
    └── .tinct/
        ├── project.yaml         # project config
        ├── keys/                # Ed25519 keys for ship signing
        ├── evidence/            # manifests + signed reports
        ├── runs/<name>/         # isolated per-run artifacts
        └── cache/               # models, datasets, chunks

Everything under ``.tinct`` is local-first and git-ignored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from tinct.core.config import ProjectConfig, dump_config, load_config
from tinct.storage.paths import TinctPaths

# Model-family gate. tinct currently supports Llama only.
SUPPORTED_MODEL_FAMILIES = ("llama",)

STATE_FILENAME = "state.json"


class UnsupportedModelFamily(ValueError):
    """Raised when a model outside the supported families is requested."""


def detect_model_family(model: str) -> str:
    """Return the (lowercased) family name for ``model``, or raise.

    Heuristic: a ``llama`` substring (or a known local path name) maps to
    ``llama``. Anything else is rejected until Qwen/DeepSeek support lands.
    """
    name = model.rsplit("/", 1)[-1].lower()
    if "llama" in name:
        return "llama"
    raise UnsupportedModelFamily(
        f"Model {model!r} is not yet supported. tinct V0 supports Llama "
        "family models only; Qwen and DeepSeek are roadmap items (see docs/ROADMAP_V1_TO_V3.md)."
    )


class Project:
    """A single tinct project on disk."""

    def __init__(self, root: Path, config: ProjectConfig,
                 paths: Optional[TinctPaths] = None) -> None:
        self.paths = paths or TinctPaths(root)
        self.root = self.paths.project_root
        self.config = config

    # -- constructors -------------------------------------------------------

    @classmethod
    def create(cls, root: Path, project_name: str, model: str,
               config: Optional[ProjectConfig] = None) -> "Project":
        """Scaffold a brand-new project, fail-closed if it already exists."""
        root = Path(root).resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(
                f"Directory {root} is not empty; refuse to scaffold over it."
            )
        cfg = config or ProjectConfig(project_name=project_name, train={"model": model})
        project = cls(root, cfg)
        project._scaffold_dirs()
        dump_config(cfg, project.config_path)
        project._write_state()
        return project

    @classmethod
    def open(cls, root: Path) -> "Project":
        """Load an existing project, failing loudly if uninitialized."""
        root = Path(root).resolve()
        paths = TinctPaths(root)
        cfg = load_config(paths.project_config)
        project = cls(root, cfg, paths=paths)
        if not paths.tinct_dir.is_dir():
            project._scaffold_dirs()
        return project

    # -- scaffolding --------------------------------------------------------

    def _scaffold_dirs(self) -> None:
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        self.paths.ensure_dirs()

    def _write_state(self) -> None:
        state = {
            "project_name": self.config.project_name,
            "model": self.config.train.model,
            "family": detect_model_family(self.config.train.model),
            "created_at": self.config.created_at,
        }
        self.state_path.write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )

    def save_config(self) -> None:
        dump_config(self.config, self.config_path)

    # -- paths (single source of truth: TinctPaths) --------------------------

    @property
    def config_path(self) -> Path:
        return self.paths.project_config

    @property
    def state_dir(self) -> Path:
        return self.paths.tinct_dir

    @property
    def state_path(self) -> Path:
        return self.state_dir / STATE_FILENAME

    @property
    def keys_dir(self) -> Path:
        return self.paths.keys_dir

    @property
    def evidence_dir(self) -> Path:
        return self.paths.evidence_dir

    @property
    def runs_dir(self) -> Path:
        return self.paths.runs_dir

    def run_dir(self, name: str) -> Path:
        return self.runs_dir / name

    def latest_run_dir(self) -> Optional[Path]:
        """Return the most recently modified run dir, or None."""
        if not self.runs_dir.is_dir():
            return None
        runs = sorted(self.runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        return runs[-1] if runs else None

    # -- helpers ------------------------------------------------------------

    def refuse_if_unsupported_model(self) -> None:
        """Fail-closed gate before any train/eval work."""
        family = detect_model_family(self.config.train.model)
        allowed = self.config.model_families_allowed
        if allowed and family not in allowed:
            raise UnsupportedModelFamily(
                f"Model family {family!r} is not among the allowed families "
                f"configured in project.yaml: {allowed}"
            )

    def create_run(self, name: str) -> Path:
        run_dir = self.run_dir(name)
        if run_dir.exists():
            raise FileExistsError(f"Run {name!r} already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        return run_dir
