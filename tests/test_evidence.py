"""Tests for evidence hashing and Ed25519 signing."""

import json
from pathlib import Path

from tinct.security.evidence import EvidenceReport, hash_path, sha256_file
from tinct.security.signing import SigningKey, verify_signature


def _sample_report():
    return EvidenceReport(
        project_name="demo",
        model="meta-llama/Llama-3.1-8B",
        family="llama",
        decision="SHIP",
        artifacts={"config": {"path": "tinct.yaml", "sha256": "abc"}},
        data_report={"title": "Data Doctor", "passed": True},
        eval_report={"title": "Eval Gate", "passed": True},
    )


def test_hash_file_roundtrip(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    h1 = sha256_file(f)
    h2 = sha256_file(f)
    assert h1 == h2
    assert len(h1) == 64


def test_hash_directory_lists_files(tmp_path: Path):
    d = tmp_path / "adapter"
    d.mkdir()
    f = d / "model.safetensors"
    f.write_text("data")
    desc = hash_path(d)
    assert desc["dir"] is True
    assert desc["files"][0]["path"].endswith("model.safetensors")
    assert len(desc["files"][0]["sha256"]) == 64


def test_sign_and_verify_roundtrip():
    key = SigningKey.generate("ship")
    report = _sample_report()
    sig = report.sign(key)
    assert isinstance(sig, bytes)
    assert report.verify()


def test_tamper_detected():
    key = SigningKey.generate("ship")
    report = _sample_report()
    report.sign(key)
    report.decision = "DON'T_SHIP"  # tamper after signing
    assert not report.verify()


def test_write_and_load_roundtrip(tmp_path: Path):
    key = SigningKey.generate("ship")
    report = _sample_report()
    report.sign(key)
    path = report.write(tmp_path, "run_1")
    loaded = EvidenceReport.load(path)
    assert loaded.project_name == "demo"
    assert loaded.verify()


def test_signature_verify_function():
    key = SigningKey.generate("k")
    data = b"payload"
    sig = key.sign_bytes(data)
    pub_pem = key.public_pem()
    assert verify_signature(pub_pem, data, sig) is True
    assert verify_signature(pub_pem, b"other", sig) is False


def test_key_save_and_load(tmp_path: Path):
    key = SigningKey.generate("ship")
    key.save(tmp_path)
    loaded = SigningKey.load(tmp_path, "ship")
    data = b"x"
    assert verify_signature(loaded.public_pem(), data, loaded.sign_bytes(data))
    with __import__("pytest").raises(FileNotFoundError):
        SigningKey.load(tmp_path, "missing")
