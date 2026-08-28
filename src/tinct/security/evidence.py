"""Cryptographic evidence for shipped checkpoints.

An :class:`EvidenceReport` collects every artifact that matters — dataset hash,
config, train metrics, eval gate report, decision — into a canonical JSON
manifest whose byte representation is **signed** with an Ed25519 key. This gives
a tamper-evident, audit-friendly record of why a model was (or wasn't) shipped.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tinct.security.signing import SigningKey, verify_signature


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_path(path: Path) -> Optional[Dict[str, Any]]:
    """Return a hashed artifact descriptor, or None if the path is missing."""
    if path.is_file():
        return {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}
    if path.is_dir():
        files = sorted([p for p in path.rglob("*") if p.is_file()])
        return {"path": str(path), "dir": True, "files": [{"path": str(p), "sha256": sha256_file(p)} for p in files]}
    return None


def hash_directory(path: Path) -> str:
    """Deterministic single sha256 over a whole directory tree.

    Hashes each file in sorted order, mixing in the file name so content alone
    cannot produce collisions between differently-named files. Returns a hex
    digest proving exactly which weights are shipping.
    """
    h = hashlib.sha256()
    files = sorted([p for p in path.rglob("*") if p.is_file()])
    for file in files:
        h.update(file.name.encode("utf-8"))
        with file.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


SIGNATURE_ALG = "ed25519"


@dataclass
class EvidenceReport:
    """A signed, tamper-evident record used to decide SHIP / DON'T-SHIP."""

    project_name: str
    model: str
    family: str
    decision: str  # "SHIP" | "DON'T_SHIP"
    created_at: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )
    artifacts: Dict[str, Any] = field(default_factory=dict)
    data_report: Dict[str, Any] = field(default_factory=dict)
    eval_report: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    safety_gates: Dict[str, Any] = field(default_factory=dict)
    # Integration-layer provenance: which tool trained the adapter, and
    # whether tinct itself executed the training run.
    training_tool: str = "tinct"
    training_executed: bool = True
    signature: Optional[Dict[str, Any]] = None

    # -- signing ------------------------------------------------------------

    def canonical_bytes(self) -> bytes:
        """Deterministic bytes over the unsigned payload (sorted keys)."""
        payload = {
            "project_name": self.project_name,
            "model": self.model,
            "family": self.family,
            "decision": self.decision,
            "created_at": self.created_at,
            "artifacts": self.artifacts,
            "data_report": self.data_report,
            "eval_report": self.eval_report,
            "metrics": self.metrics,
            "config": self.config,
            "safety_gates": self.safety_gates,
            "training_tool": self.training_tool,
            "training_executed": self.training_executed,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def sign(self, key: SigningKey) -> bytes:
        sig = key.sign_bytes(self.canonical_bytes())
        self.signature = {
            "alg": SIGNATURE_ALG,
            "key_name": key.name,
            "public_key_pem": key.public_pem().decode("ascii"),
            "value": sig.hex(),
        }
        return sig

    def verify(self) -> bool:
        """Verify our own signature if present."""
        if not self.signature:
            return False
        public_pem = self.signature["public_key_pem"].encode("ascii")
        sig = bytes.fromhex(self.signature["value"])
        return verify_signature(public_pem, self.canonical_bytes(), sig)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "model": self.model,
            "family": self.family,
            "decision": self.decision,
            "created_at": self.created_at,
            "artifacts": self.artifacts,
            "data_report": self.data_report,
            "eval_report": self.eval_report,
            "metrics": self.metrics,
            "config": self.config,
            "safety_gates": self.safety_gates,
            "training_tool": self.training_tool,
            "training_executed": self.training_executed,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EvidenceReport":
        return cls(
            project_name=raw["project_name"],
            model=raw["model"],
            family=raw["family"],
            decision=raw["decision"],
            created_at=raw.get("created_at", ""),
            artifacts=raw.get("artifacts", {}),
            data_report=raw.get("data_report", {}),
            eval_report=raw.get("eval_report", {}),
            metrics=raw.get("metrics", {}),
            config=raw.get("config", {}),
            safety_gates=raw.get("safety_gates", {}),
            training_tool=raw.get("training_tool", "tinct"),
            training_executed=raw.get("training_executed", True),
            signature=raw.get("signature"),
        )

    def write(self, evidence_dir: Path, name: str) -> Path:
        path = evidence_dir / f"{name}_evidence.json"
        dump_json(self.to_dict(), path)
        return path

    @classmethod
    def load(cls, path: Path) -> "EvidenceReport":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)


def evidence_filename(run_name: str) -> str:
    return f"{run_name}_evidence.json"
