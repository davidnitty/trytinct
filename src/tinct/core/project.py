"""Project lifecycle and on-disk state layout.

A tinct project root looks like::

    project/
    ├── tinct.yaml          # ProjectConfig (see core/config.py)
    ├── data/               # user datasets (unmanaged)
    ├── runs/<name>/        # per-run artifacts: trainer_state.json, metrics.json, adapter/
    └── .tinct/
        ├── keys/           # Ed25519 keys for ship signing
        ├── evidence/       # manifests + signed reports
        └── state.json      # lightweight bookkeeping

All of this is local-first. The ``.tinct`` folder is git-ignored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from tinct.core.config import (
    CONFIG_FILENAME,
    ProjectConfig,
    dump_config,
    load_config,
)

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

    def __init__(self, root: Path, config: ProjectConfig) -> None:
        self.root = root
        self.config = config

    # -- constructors -------------------------------------------------------

    @classmethod
    def create(cls, root: Path, project_name: str, model: str,
               config: Optional[ProjectConfig] = None) -> "Project":
        """Scaffold a brand-new project, fail-closed if it already exists."""
        root = root.resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(
                f"Directory {root} is not empty; refuse to scaffold over it."
            )
        cfg = config or ProjectConfig(project_name=project_name, train={"model": model})
        project = cls(root, cfg)
        project._scaffold_dirs()
        dump_config(cfg, root / CONFIG_FILENAME)
        project._write_state()
        return project

    @classmethod
    def open(cls, root: Path) -> "Project":
        """Load an existing project, failing loudly if uninitialized."""
        root = root.resolve()
        cfg_path = root / CONFIG_FILENAME
        cfg = load_config(cfg_path)
        project = cls(root, cfg)
        if not (root / ".tinct").is_dir():
            project._scaffold_dirs()
        return project

    # -- scaffolding --------------------------------------------------------

    def _scaffold_dirs(self) -> None:
        for d in ("data", "runs", self.state_dir, self.keys_dir, self.evidence_dir):
            (self.root / d).mkdir(parents=True, exist_ok=True)

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
        dump_config(self.config, self.root / CONFIG_FILENAME)

    # -- paths --------------------------------------------------------------

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_FILENAME

    @property
    def state_dir(self) -> Path:
        return self.root / ".tinct"

    @property
    def state_path(self) -> Path:
        return self.state_dir / STATE_FILENAME

    @property
    def keys_dir(self) -> Path:
        return self.state_dir / "keys"

    @property
    def evidence_dir(self) -> Path:
        return self.state_dir / "evidence"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

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
        detect_model_family(self.config.train.model)

    def create_run(self, name: str) -> Path:
        run_dir = self.run_dir(name)
        if run_dir.exists():
            raise FileExistsError(f"Run {name!r} already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        return run_dir
