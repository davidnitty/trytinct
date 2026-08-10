"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinct.core.project import Project


def make_dataset(path: Path, n: int = 40, good: bool = True) -> Path:
    rows = []
    for i in range(n):
        if good:
            rows.append({"instruction": f"Tell me about item {i}", "output": f"Item {i} is great."})
        elif i == 0:
            rows.append({})  # missing columns
        elif i == 1:
            rows.append({"instruction": "", "output": "empty instruction"})
        else:
            rows.append({"instruction": f"Q{i}", "output": f"A{i}"})
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    return make_dataset(tmp_path / "data.jsonl")


@pytest.fixture
def bad_dataset(tmp_path: Path) -> Path:
    return make_dataset(tmp_path / "bad.jsonl", good=False)


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(
        tmp_path / "proj",
        project_name="demo",
        model="meta-llama/Llama-3.1-8B",
    )
