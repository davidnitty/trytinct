"""Data Doctor — validates instruction datasets before training.

Fail-closed: hard (error-severity) failures block training. Warnings and info
are reported but do not block.

Data is read from JSON or JSONL files containing a list/stream of
instruction records, shaped per :class:`tinct.core.config.DataConfig`.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tinct.core.config import DataConfig
from tinct.core.rules import RuleReport, Severity, error, info, ok, warn


class DatasetLoadError(ValueError):
    """Raised when a dataset file cannot be read or parsed."""


APPROX_CHARS_PER_TOKEN = 4


class DataDoctor:
    """Stateless validator. Each :meth:`run` produces a fresh report."""

    def __init__(self, config: DataConfig, max_seq_len: int = 2048, seed: int = 42) -> None:
        self.config = config
        self.max_seq_len = max_seq_len
        self.seed = seed

    # -- loading ------------------------------------------------------------

    def load(self, path: Path) -> List[Dict[str, Any]]:
        """Load records from a JSON or JSONL file."""
        path = Path(path)
        if not path.is_file():
            raise DatasetLoadError(f"Dataset not found: {path}")
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        try:
            if suffix == ".jsonl" or suffix == ".ndjson":
                records = [json.loads(line) for line in text.splitlines() if line.strip()]
            elif suffix == ".json":
                data = json.loads(text)
                records = data if isinstance(data, list) else list(data.values())
            else:
                # Fall back: try JSONL first, then JSON.
                try:
                    records = [json.loads(line) for line in text.splitlines() if line.strip()]
                except json.JSONDecodeError:
                    data = json.loads(text)
                    records = data if isinstance(data, list) else list(data.values())
        except json.JSONDecodeError as exc:
            raise DatasetLoadError(f"Could not parse dataset {path}: {exc}") from exc

        return [r for r in records if isinstance(r, dict)]

    # -- rules --------------------------------------------------------------

    def run(self, path: Path) -> Tuple[RuleReport, List[Dict[str, Any]]]:
        """Validate the dataset and return (report, records)."""
        report = RuleReport("Data Doctor")
        try:
            records = self.load(path)
        except DatasetLoadError as exc:
            report.add(error("data.readable", "Dataset readable", str(exc)))
            return report, []

        report.add(ok("data.readable", "Dataset readable", f"{len(records)} rows loaded"))

        # Not empty / min rows (fail-closed).
        if len(records) == 0:
            report.add(error("data.not_empty", "Dataset not empty",
                             "Dataset contains zero records."))
            return report, []

        report.add(
            ok("data.min_rows", "Minimum row count",
               f"{len(records)} >= minimum {self.config.min_rows}")
            if len(records) >= self.config.min_rows
            else error("data.min_rows", "Minimum row count",
                       f"{len(records)} rows is below the fail-closed minimum {self.config.min_rows}")
        )

        self._check_columns(report, records)
        self._check_empty_fields(report, records)
        self._check_duplicates(report, records)
        self._check_lengths(report, records)
        self._check_token_estimate(report, records)
        self._check_output_diversity(report, records)

        return report, records

    # -- individual checks --------------------------------------------------

    def _check_columns(self, report: RuleReport, records: List[Dict[str, Any]]) -> None:
        cfg = self.config
        required = {
            cfg.instruction_column: "instruction column",
            cfg.output_column: "output column",
        }
        missing = [name for name, _ in required.items() if name not in records[0]]
        if missing:
            report.add(error(
                "data.columns",
                "Required columns present",
                f"Missing required columns {missing} in the first record "
                f"(expected {list(required)}).",
                meta={"missing": missing},
            ))
            return
        report.add(ok("data.columns", "Required columns present"))

    def _check_empty_fields(self, report: RuleReport, records: List[Dict[str, Any]]) -> None:
        cfg = self.config
        empty_instr = []
        empty_out = []
        for i, rec in enumerate(records):
            if not str(rec.get(cfg.instruction_column, "") or "").strip():
                empty_instr.append(i)
            if not str(rec.get(cfg.output_column, "") or "").strip():
                empty_out.append(i)
        total = len(records)
        if empty_instr:
            report.add(error(
                "data.empty_instruction",
                "Instruction fields populated",
                f"{len(empty_instr)}/{total} rows have an empty instruction.",
                meta={"row_indices": empty_instr[:20]},
            ))
        else:
            report.add(ok("data.empty_instruction", "Instruction fields populated"))
        if empty_out:
            report.add(error(
                "data.empty_output",
                "Output fields populated",
                f"{len(empty_out)}/{total} rows have an empty output.",
                meta={"row_indices": empty_out[:20]},
            ))
        else:
            report.add(ok("data.empty_output", "Output fields populated"))

    def _check_duplicates(self, report: RuleReport, records: List[Dict[str, Any]]) -> None:
        seen = set()
        dups = 0
        for rec in records:
            key = json.dumps(rec, sort_keys=True, ensure_ascii=False)
            if key in seen:
                dups += 1
            seen.add(key)
        if dups:
            report.add(warn(
                "data.duplicates", "Duplicate rows",
                f"{dups} exact duplicate row(s) detected.",
                meta={"duplicates": dups},
            ))
        else:
            report.add(ok("data.duplicates", "Duplicate rows", "No exact duplicates"))

    def _check_lengths(self, report: RuleReport, records: List[Dict[str, Any]]) -> None:
        cfg = self.config
        lengths = [
            len(str(rec.get(cfg.instruction_column, ""))) + len(str(rec.get(cfg.output_column, "")))
            for rec in records
        ]
        n = len(lengths)
        mean = statistics.mean(lengths)
        longest = max(lengths)
        report.add(info(
            "data.length_stats", "Row length statistics",
            f"mean {mean:.1f} chars, longest {longest} chars across {n} rows.",
            meta={"mean_chars": round(mean, 1), "max_chars": longest, "n": n},
        ))

    def _check_token_estimate(self, report: RuleReport, records: List[Dict[str, Any]]) -> None:
        cfg = self.config
        max_tokens = self.max_seq_len
        worst = 0
        for rec in records:
            text = str(rec.get(cfg.instruction_column, "")) + str(rec.get(cfg.output_column, ""))
            est = len(text) // APPROX_CHARS_PER_TOKEN
            worst = max(worst, est)
        if worst > max_tokens:
            report.add(warn(
                "data.token_estimate", "Token-length estimate",
                f"Longest row ~{worst} tokens may exceed context max_seq_len={max_tokens}; "
                "truncation will apply.",
                meta={"estimated_max_tokens": worst, "max_seq_len": max_tokens},
            ))
        else:
            report.add(ok("data.token_estimate", "Token-length estimate",
                          f"Estimated max {worst} tokens <= {max_tokens}"))

    def _check_output_diversity(self, report: RuleReport, records: List[Dict[str, Any]]) -> None:
        cfg = self.config
        outputs = [str(rec.get(cfg.output_column, "")) for rec in records]
        unique = len({o for o in outputs if o.strip()})
        total = len(outputs)
        ratio = unique / total if total else 0.0
        report.add(info(
            "data.output_diversity", "Output diversity",
            f"{unique} unique outputs out of {total} rows ({ratio:.0%}).",
            meta={"unique_outputs": unique, "total": total},
        ))

    # -- deterministic split ------------------------------------------------

    def split(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Deterministic train/validation split honoring data ratios.

        Returns (train_records, valid_records) with an approximate 80/20
        reserve when only a 90/10 ratio is configured.
        """
        cfg = self.config
        max_samples = cfg.max_samples or len(records)
        items = list(records[:max_samples])
        rng = random.Random(self.seed)
        rng.shuffle(items)

        n = len(items)
        train_ratio = cfg.train_ratio
        train_n = int(round(n * train_ratio))
        # Leave a validation slice even if the split math is tight.
        split_at = max(1, min(train_n, n - 1)) if n > 2 else n
        return items[:split_at], items[split_at:]


# Convenience -----------------------------------------------------------------

def validate_dataset(path: Path, config: DataConfig) -> Tuple[RuleReport, List[Dict[str, Any]]]:
    return DataDoctor(config).run(path)
