"""Tests for the streaming inference engine (no heavy ML deps required)."""

import pytest

from tinct.engine.deps import MissingDependencyError
from tinct.engine.streaming import StreamingInferenceEngine


def test_import_does_not_require_torch():
    # Guarded import passes even though torch/transformers are not installed.
    import tinct.engine.streaming  # noqa: F401


def test_instantiate_cpu_default(tmp_path):
    engine = StreamingInferenceEngine(model_path=tmp_path, chunk_dir=tmp_path)
    assert engine.device == "cpu"  # resolved before any cuda probe


def test_stream_forward_fails_closed(tmp_path):
    engine = StreamingInferenceEngine(model_path=tmp_path, chunk_dir=tmp_path)
    with pytest.raises(NotImplementedError):
        engine.stream_forward(None)


def test_cleanup_does_not_raise_without_model(tmp_path):
    engine = StreamingInferenceEngine(model_path=tmp_path, chunk_dir=tmp_path)
    engine.cleanup()  # should be a no-op, no error


def test_load_meta_model_guards_missing_deps(tmp_path, monkeypatch):
    # Prove the dependency guard fires before any heavy work — regardless of
    # whether torch happens to be installed in this environment.
    import tinct.engine.streaming as streaming
    monkeypatch.setattr(streaming, "ensure_train_deps",
                        lambda **kw: (_ for _ in ()).throw(MissingDependencyError("train")))
    engine = StreamingInferenceEngine(model_path=tmp_path, chunk_dir=tmp_path)
    with pytest.raises(MissingDependencyError):
        engine.load_meta_model()
