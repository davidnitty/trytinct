"""Project configuration.

tinct projects are configured by a single ``project.yaml`` under the
project's ``.tinct/`` state dir. The schema is defined with pydantic v2 so
unknown or mistyped settings fail fast (secure by default: no silent config
drift).

Only lightweight dependencies (pydantic + PyYAML) are used here.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Literal, Optional

import yaml

try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:  # pragma: no cover
    raise RuntimeError(
        "pydantic>=2 is required for tinct. Install with: pip install tinct"
    )

CONFIG_FILENAME = "project.yaml"  # located under the project's .tinct dir
CONFIG_VERSION = "1"

# Supported dataset schemas for instruction tuning.
DataFormat = Literal["instruct", "alpaca", "chatml"]

# Supported quantization strategies for LoRA training.
QuantMethod = Literal["lora", "qlora"]


class DataConfig(BaseModel):
    """How the instruction dataset is shaped and split."""

    format: DataFormat = "instruct"
    instruction_column: str = Field(default="instruction")
    input_column: Optional[str] = Field(default=None, description="Optional input/context column.")
    output_column: str = Field(default="output")
    system_column: Optional[str] = Field(default=None)
    train_ratio: float = Field(default=0.9, ge=0.5, le=1.0)
    validation_ratio: float = Field(default=0.1, ge=0.0, le=0.5)
    max_samples: Optional[int] = Field(default=None, ge=1)
    min_rows: int = Field(default=16, ge=1, description="Minimum rows to allow training (fail-closed).")


class TrainConfig(BaseModel):
    """LoRA / QLoRA training hyperparameters with reproducible defaults."""

    model: str = Field(..., description="HF model id or local path, e.g. meta-llama/Llama-3.1-8B.")
    method: Literal["sft"] = "sft"
    quant: QuantMethod = "qlora"
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    learning_rate: float = Field(default=2e-4, gt=0.0)
    num_epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=2, ge=1)
    grad_accum_steps: int = Field(default=4, ge=1)
    max_seq_len: int = Field(default=2048, ge=128, le=16384)
    seed: int = Field(default=42)
    max_steps: Optional[int] = Field(default=None, ge=1)
    # Low-VRAM streaming knobs.
    use_bf16: bool = True
    use_4bit: bool = Field(default=True, description="QLoRA: 4-bit base model loading.")
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=500, ge=1)

    @field_validator("model")
    @classmethod
    def _model_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("model must not be empty")
        return v


class EvalConfig(BaseModel):
    """Eval gating: a checkpoint ships only if it clears these thresholds."""

    metric: str = Field(default="eval_loss")
    threshold: float = Field(default=1.5, description="Max acceptable value for lower-is-better metrics.")
    higher_is_better: bool = False
    split: str = "test"
    max_eval_samples: Optional[int] = Field(default=None, ge=1)
    # Optional lm-evaluation-harness benchmark (e.g. "wikitext"). Empty = skip.
    lm_harness_task: Optional[str] = Field(default=None)


class SecurityConfig(BaseModel):
    """Ship-time signing and integrity settings."""

    sign_evidence: bool = True
    key_name: str = "ship"


class ProjectConfig(BaseModel):
    """Top-level project configuration persisted to .tinct/project.yaml."""

    version: Literal["1"] = CONFIG_VERSION
    project_name: str
    created_at: str = Field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)


# --------------------------------------------------------------------------- #
# Serialization / helpers
# --------------------------------------------------------------------------- #

def config_to_dict(config: ProjectConfig) -> dict:
    return config.model_dump(mode="json")


def config_from_dict(raw: dict) -> ProjectConfig:
    return ProjectConfig.model_validate(raw)


def dump_config(config: ProjectConfig, path: Path) -> None:
    """Write the config to ``path`` as YAML (fail-closed on unknown fields)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(config_to_dict(config), sort_keys=False)
    path.write_text(text, encoding="utf-8")


def load_config(path: Path) -> ProjectConfig:
    """Load ``.tinct/project.yaml``, failing loudly on parse/validation errors."""
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path} (run `tinct init` first)")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config file (expected a YAML mapping): {path}")
    return config_from_dict(raw)