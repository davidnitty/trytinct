"""Tests for the automatic model-prep step wired into ``tinct train``."""

import json
from pathlib import Path

from tinct.cli.train_cmd import prepare_base_model_chunks
from tinct.core.project import Project


def _fake_model(model_dir: Path):
    model_dir.mkdir(parents=True)
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"shard-a")
    (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"shard-b")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}})
    )
    return model_dir


def test_prepare_chunks_manifest_into_run(tmp_path: Path):
    model_dir = _fake_model(tmp_path / "model")
    project = Project.create(tmp_path / "proj", "demo", "meta-llama/Llama-3.1-8B")
    run_dir = project.create_run("run_x")

    resolved = prepare_base_model_chunks(project, run_dir, str(model_dir))

    assert resolved.is_dir()
    manifest_path = run_dir / "base_model_chunks.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest) == 2
    assert all(len(h) == 64 for h in manifest.values())

    # Chunks are cached under the project's .tinct dir (git-ignored).
    cached = tmp_path / "proj" / ".tinct" / "cache" / "chunks"
    assert (cached / "model-00001-of-00002.safetensors").is_file()