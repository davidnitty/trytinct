# Generate a genuine signed evidence bundle for frontend development.
# Uses tinct's own EvidenceReport + SigningKey so the signature is real and
# the frontend's server-side Ed25519 verification is exercised end-to-end.
from pathlib import Path

from tinct.security.evidence import EvidenceReport
from tinct.security.signing import SigningKey

PROJECT = Path(__file__).parent
KEYS = PROJECT / ".tinct" / "keys"
EVIDENCE = PROJECT / ".tinct" / "evidence"

key_path = KEYS / "default_private.pem"
if key_path.is_file():
    key = SigningKey.load(KEYS, "default")
else:
    key = SigningKey.generate("default")
    key.save(KEYS)

safety_gates = {
    "result": "PASS",
    "canary_leakage": {
        "status": "PASS",
        "canaries_tested": 50,
        "canaries_leaked": 0,
        "leakage_rate": 0.0,
        "threshold": 0.5,
    },
    "refusal_regression": {
        "status": "PASS",
        "base_refusal_rate": 0.85,
        "adapter_refusal_rate": 0.88,
        "regression_rate": -0.035,
        "threshold": 0.2,
    },
    "toxicity": {
        "status": "PASS",
        "method": "heuristic",
        "base_toxicity_avg": 0.02,
        "adapter_toxicity_avg": 0.018,
        "increase_factor": 0.9,
        "threshold": 2.0,
    },
    "expert_collapse": {
        "status": "PASS",
        "num_experts": 8,
        "total_tokens_routed": 5120,
        "min_utilization": 0.11,
        "max_utilization": 0.14,
        "laziest_expert_id": 2,
        "threshold": 0.01,
    },
    "routing_regression": {
        "status": "PASS",
        "num_experts": 8,
        "regressed_experts": [],
        "base_utilization": [0.125, 0.122, 0.13, 0.128, 0.119, 0.124, 0.126, 0.126],
        "adapter_utilization": [0.131, 0.128, 0.115, 0.12, 0.135, 0.122, 0.118, 0.131],
        "adapter_over_base_ratio": [1.048, 1.049, 0.885, 0.938, 1.134, 0.984, 0.937, 1.04],
        "relative_drop_threshold": 0.5,
        "base_floor": 0.02,
    },
    "offload_stats": {
        "h2d_streams": 142,
        "d2h_evictions": 134,
        "cache_hits": 8450,
        "bytes_h2d": 4294967296,
        "bytes_d2h": 4180000000,
    },
}

report = EvidenceReport(
    project_name="demo-cert",
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    family="mistral",
    decision="SHIP",
    artifacts={
        "adapter": {
            "path": "runs/cert_20260904_143022/adapter",
            "dir": True,
            "files": [
                {"path": "runs/cert_20260904_143022/adapter/adapter_model.safetensors", "sha256": "9f2c02a71b4d1e5f0c8a6d3e7b9c1a4f2e8d6b0c3a5f7e9d1c3b5a7f9e1d3c5b"},
                {"path": "runs/cert_20260904_143022/adapter/adapter_config.json", "sha256": "1a2b3c4d5e6f70819a2b3c4d5e6f70819a2b3c4d5e6f70819a2b3c4d5e6f7081a"},
            ],
            "adapter_sha256": "4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b",
        },
        "dataset": {"path": "examples/mistral_data.jsonl", "sha256": "7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d", "size": 18432},
    },
    data_report={"format": "instruct", "rows": 16, "errors": 0, "model_family": "mistral"},
    eval_report={"gate": "generation_smoke_test", "status": "PASS", "empty_responses": 0, "repetitive_responses": 0},
    metrics={"final_loss": 1.42, "steps": 24},
    config={"method": "certify", "lora_r": 16, "max_seq_len": 2048},
    safety_gates=safety_gates,
    training_tool="unsloth",
    training_executed=False,
)
report.sign(key)
path = report.write(EVIDENCE, "cert_20260904_143022")
print("verified:", report.verify())
print("wrote:", path)
