"""Tests for project scaffolding, config, and the Data Doctor."""

import json
from pathlib import Path

import pytest

from tinct.core.config import ProjectConfig, load_config
from tinct.core.datadoctor import (
    DataDoctor,
    DatasetLoadError,
    check_dataset_format,
    validate_dpo_row,
)
from tinct.core.project import Project, UnsupportedModelFamily, detect_model_family


# -- project ---------------------------------------------------------------

def test_project_create_and_reopen(tmp_path: Path):
    project = Project.create(tmp_path / "p", "demo", "meta-llama/Llama-3.1-8B")
    assert project.config_path.is_file()
    assert (tmp_path / "p" / ".tinct" / "keys").is_dir()

    reopened = Project.open(tmp_path / "p")
    assert reopened.config.project_name == "demo"
    assert reopened.config.train.model == "meta-llama/Llama-3.1-8B"


def test_project_create_refuses_nonempty(tmp_path: Path):
    d = tmp_path / "p"
    d.mkdir()
    (d / "file.txt").write_text("x")
    with pytest.raises(FileExistsError):
        Project.create(d, "demo", "meta-llama/Llama-3.1-8B")


def test_unsupported_model_blocks(tmp_path: Path):
    with pytest.raises(UnsupportedModelFamily):
        detect_model_family("unknown/arch-9000")
    p = Project.create(tmp_path / "p", "demo", "meta-llama/Llama-3.1-8B")
    p.config.train.model = "deepseek-ai/DeepSeek-R1"
    # Detected as deepseek, but not allowed by default -> fail-closed.
    with pytest.raises(UnsupportedModelFamily):
        p.refuse_if_unsupported_model()


def test_config_roundtrip(tmp_path: Path):
    cfg = ProjectConfig(project_name="x", train={"model": "meta-llama/Llama-3.1-8B"})
    from tinct.core.config import dump_config
    path = tmp_path / "tinct.yaml"
    dump_config(cfg, path)
    loaded = load_config(path)
    assert loaded.train.model == cfg.train.model
    assert loaded.eval.threshold == 1.5


# -- data doctor -----------------------------------------------------------

def test_valid_dataset_passes(project, sample_dataset: Path):
    doctor = DataDoctor(project.config.data, max_seq_len=2048, seed=42)
    report, records = doctor.run(sample_dataset)
    assert report.passed
    assert len(records) == 40


def test_invalid_dataset_fails(project, bad_dataset: Path):
    doctor = DataDoctor(project.config.data, max_seq_len=2048, seed=42)
    report, _ = doctor.run(bad_dataset)
    assert not report.passed
    ids = {r.rule_id for r in report.failed_errors}
    assert "data.columns" in ids
    assert "data.empty_instruction" in ids


def test_min_rows_fail_closed(project, tmp_path: Path):
    data = tmp_path / "tiny.jsonl"
    data.write_text(json.dumps({"instruction": "a", "output": "b"}) + "\n")
    doctor = DataDoctor(project.config.data, max_seq_len=2048, seed=42)
    report, _ = doctor.run(data)
    assert not report.passed
    assert any(r.rule_id == "data.min_rows" for r in report.failed_errors)


def test_missing_file_raises(project, tmp_path: Path):
    doctor = DataDoctor(project.config.data, max_seq_len=2048, seed=42)
    with pytest.raises(DatasetLoadError):
        doctor.load(tmp_path / "nope.jsonl")


def test_missing_file_reports_error(project, tmp_path: Path):
    doctor = DataDoctor(project.config.data, max_seq_len=2048, seed=42)
    report, _ = doctor.run(tmp_path / "nope.jsonl")
    assert not report.passed
    assert any(r.rule_id == "data.readable" for r in report.failed_errors)


def test_deterministic_split(project, sample_dataset: Path):
    doctor = DataDoctor(project.config.data, max_seq_len=2048, seed=42)
    _, records = doctor.run(sample_dataset)
    a_train, a_valid = doctor.split(records)
    # same seed -> same split
    b_train, b_valid = doctor.split(records)
    assert a_train == b_train
    assert a_valid == b_valid
    assert len(a_train) + len(a_valid) == len(records)


def test_text_format_dataset_auto_detected(project, tmp_path: Path):
    data = tmp_path / "text.jsonl"
    lines = [
        {"text": "<|begin_of_text|>user: q<|eot_id|>assistant: a"},
        {"text": "<|begin_of_text|>user: q2<|eot_id|>assistant: a2"},
    ]
    lines += [{"text": f"row {i} text"} for i in range(20)]
    data.write_text("\n".join(json.dumps(r) for r in lines), encoding="utf-8")

    doctor = DataDoctor(project.config.data, max_seq_len=2048, seed=42)
    report, records = doctor.run(data)
    assert report.passed
    assert doctor.format_used == "text"
    assert len(records) == 22


# -- DPO data ---------------------------------------------------------------

def test_check_dataset_format_detects_dpo_and_sft():
    assert check_dataset_format({"prompt": "p", "chosen": "a", "rejected": "b"}) == "dpo"
    assert check_dataset_format({"text": "hello"}) == "sft"
    assert check_dataset_format({"instruction": "q", "output": "a"}) == "unknown"


def test_validate_dpo_row_ok():
    row = {"prompt": "p", "chosen": "good answer", "rejected": "bad answer"}
    assert validate_dpo_row(row, 0) == []


def test_validate_dpo_row_missing_keys():
    errors = validate_dpo_row({"prompt": "p", "chosen": "a"}, 3)
    assert len(errors) == 1
    assert "Row 3" in errors[0]
    assert "Missing keys" in errors[0]


def test_validate_dpo_row_empty_or_non_string():
    errors = validate_dpo_row({"prompt": "p", "chosen": "", "rejected": "b"}, 1)
    assert any("chosen" in e and "empty" in e for e in errors)
    errors2 = validate_dpo_row({"prompt": "p", "chosen": 42, "rejected": "b"}, 1)
    assert any("chosen" in e and "not a string" in e for e in errors2)


def test_validate_dpo_row_chosen_equals_rejected():
    errors = validate_dpo_row({"prompt": "p", "chosen": "same", "rejected": "same"}, 0)
    assert any("identical" in e for e in errors)


def test_dpo_dataset_passes_doctor(project, tmp_path: Path):
    data = tmp_path / "dpo.jsonl"
    rows = [{"prompt": f"Question {i}?", "chosen": f"Good answer {i}",
             "rejected": f"Bad answer {i}"} for i in range(20)]
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    doctor = DataDoctor(project.config.data, max_seq_len=2048, seed=42)
    report, records = doctor.run(data)
    assert report.passed
    assert doctor.format_used == "dpo"
    assert len(records) == 20


def test_dpo_dataset_with_identical_pairs_fails(project, tmp_path: Path):
    data = tmp_path / "dpo.jsonl"
    rows = [{"prompt": f"Q{i}", "chosen": "same", "rejected": "same"} for i in range(20)]
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    doctor = DataDoctor(project.config.data, max_seq_len=2048, seed=42)
    report, _ = doctor.run(data)
    assert not report.passed
    assert any(r.rule_id == "data.dpo.distinct" for r in report.failed_errors)
