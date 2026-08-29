"""Tests for storage paths (cache dir + model resolution)."""

import json
from pathlib import Path

import pytest

from tinct.engine.deps import MissingDependencyError
from tinct.storage.paths import TinctPaths, get_cache_dir, resolve_model


def _fake_model(model_dir: Path):
    model_dir.mkdir(parents=True)
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"a")
    (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"b")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}})
    )
    return model_dir


def test_get_cache_dir_defaults_to_cwd_relative():
    cache = get_cache_dir()
    assert cache.name == "cache"
    # project-scoped: <root>/.tinct/cache
    assert cache.parent.name == ".tinct"


def test_get_cache_dir_project_root():
    cache = get_cache_dir(Path("some/project"))
    assert str(cache) == str(Path("some/project") / ".tinct" / "cache")


def test_resolve_model_local_ok(tmp_path: Path):
    model_dir = _fake_model(tmp_path / "model")
    resolved = resolve_model(str(model_dir))
    assert resolved == model_dir.resolve()
    assert resolved.is_dir()


def test_resolve_model_blocks_missing_index(tmp_path: Path):
    bad = tmp_path / "model"
    bad.mkdir()
    (bad / "pytorch_model.bin").write_bytes(b"pickle!")
    with pytest.raises(ValueError, match="safetensors"):
        resolve_model(str(bad))


def test_resolve_hub_id_requires_huggingface_hub(monkeypatch):
    # Deterministic regardless of whether huggingface_hub is installed: force
    # the snapshot helper to raise the dep-guard error and assert it propagates.
    from tinct.storage import paths as paths_mod
    monkeypatch.setattr(
        paths_mod, "_snapshot_hub",
        lambda hub_id: (_ for _ in ()).throw(MissingDependencyError("train")),
    )
    with pytest.raises(MissingDependencyError):
        resolve_model("meta-llama/Llama-3.2-1B")


def test_tinct_paths_full_tree(tmp_path: Path):
    paths = TinctPaths(tmp_path)
    assert paths.tinct_dir == tmp_path / ".tinct"
    assert paths.project_config == tmp_path / ".tinct" / "project.yaml"
    assert paths.cache_dir == tmp_path / ".tinct" / "cache"
    assert paths.model_cache == paths.cache_dir / "models"
    assert paths.dataset_cache == paths.cache_dir / "datasets"
    assert paths.runs_dir == tmp_path / ".tinct" / "runs"
    assert paths.keys_dir == tmp_path / ".tinct" / "keys"
    assert paths.evidence_dir == tmp_path / ".tinct" / "evidence"

    paths.ensure_dirs()
    for d in (paths.cache_dir, paths.model_cache, paths.dataset_cache,
              paths.runs_dir, paths.keys_dir, paths.evidence_dir):
        assert d.is_dir()


def test_discover_walks_up_to_project(tmp_path: Path):
    paths = TinctPaths(tmp_path)
    paths.ensure_dirs()
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    # chdir into the nested dir and rediscover from there.
    import os
    prev = os.getcwd()
    try:
        os.chdir(nested)
        assert TinctPaths.discover().project_root.resolve() == tmp_path.resolve()
    finally:
        os.chdir(prev)