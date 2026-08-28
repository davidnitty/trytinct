"""Tests for project initialization (init_tinct_project)."""

from pathlib import Path

import pytest
import yaml

from tinct.core.init_project import PROJECT_GITIGNORE, init_tinct_project
from tinct.core.project import UnsupportedModelFamily
from tinct.storage.paths import TinctPaths


def test_init_creates_tree_and_config(tmp_path: Path):
    proj = init_tinct_project(tmp_path / "p", "meta-llama/Llama-3.1-8B",
                              project_name="p")
    assert proj.root == tmp_path / "p"

    # Full .tinct tree exists.
    paths = TinctPaths(tmp_path / "p")
    for d in (paths.cache_dir, paths.model_cache, paths.dataset_cache,
              paths.runs_dir, paths.keys_dir, paths.evidence_dir):
        assert d.is_dir()

    # Default project.yaml with the init policy keys.
    raw = yaml.safe_load(paths.project_config.read_text(encoding="utf-8"))
    assert raw["project_name"] == "p"
    assert raw["model_families_allowed"] == ["llama", "mistral"]
    assert raw["default_lora_rank"] == 16
    assert raw["max_loss_threshold"] == 10.0
    assert raw["fail_closed"] is True
    assert raw["train"]["model"] == "meta-llama/Llama-3.1-8B"

    # .gitignore seeded in the project root.
    gitignore = tmp_path / "p" / ".gitignore"
    assert gitignore.read_text(encoding="utf-8") == PROJECT_GITIGNORE


def test_init_is_idempotent_on_config(tmp_path: Path):
    first = init_tinct_project(tmp_path / "p", "meta-llama/Llama-3.1-8B")
    cfg_before = (tmp_path / "p" / ".tinct" / "project.yaml").read_text(encoding="utf-8")
    # Re-init only when dir still empty? Dir now has files -> should fail-closed.
    with pytest.raises(FileExistsError):
        init_tinct_project(tmp_path / "p", "meta-llama/Llama-3.1-8B")
    # Config unchanged after failed re-init.
    assert (tmp_path / "p" / ".tinct" / "project.yaml").read_text(encoding="utf-8") == cfg_before


def test_init_rejects_unsupported_family(tmp_path: Path):
    with pytest.raises(UnsupportedModelFamily):
        init_tinct_project(tmp_path / "p", "deepseek-ai/DeepSeek-R1")


def test_init_rejects_nonempty_dir(tmp_path: Path):
    d = tmp_path / "p"
    d.mkdir()
    (d / "file.txt").write_text("x")
    with pytest.raises(FileExistsError):
        init_tinct_project(d, "meta-llama/Llama-3.1-8B")
