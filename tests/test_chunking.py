"""Tests for the model chunker (airllm-style model prep)."""

import json
from pathlib import Path

import pytest

from tinct.engine.chunking import ModelChunker


def _make_model(model_dir: Path, n_shards: int = 2):
    model_dir.mkdir(parents=True)
    weight_map = {}
    for i in range(n_shards):
        name = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
        (model_dir / name).write_bytes(b"shard-data-" + str(i).encode())
        weight_map[f"tensor.{i}"] = name
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    return model_dir


def test_chunk_copies_and_hashes(tmp_path: Path):
    model_dir = _make_model(tmp_path / "model")
    cache = tmp_path / "cache"
    chunker = ModelChunker(cache)
    manifest = chunker.chunk_model(model_dir)

    assert set(manifest.keys()) == {"model-00001-of-00002.safetensors",
                                    "model-00002-of-00002.safetensors"}
    # Chunks were copied into cache.
    assert (cache / "chunks" / "model-00001-of-00002.safetensors").is_file()
    # Hashes are valid sha256 hex.
    for h in manifest.values():
        assert len(h) == 64


def test_chunk_is_reproducible(tmp_path: Path):
    model_dir = _make_model(tmp_path / "model")
    cache = tmp_path / "cache"
    m1 = ModelChunker(cache).chunk_model(model_dir)
    m2 = ModelChunker(cache).chunk_model(model_dir)
    assert m1 == m2  # deterministic manifest


def test_pickle_bin_blocked(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "pytorch_model.bin").write_bytes(b"pickle!")
    chunker = ModelChunker(tmp_path / "cache")
    with pytest.raises(ValueError, match="safetensors"):
        chunker.chunk_model(model_dir)


def test_missing_safetensors_index_blocked(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    chunker = ModelChunker(tmp_path / "cache")
    with pytest.raises(ValueError, match="safetensors"):
        chunker.chunk_model(model_dir)
