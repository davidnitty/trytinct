# Generate genuine signed evidence bundles for frontend development.
# Uses tinct's own EvidenceReport + SigningKey so the signature is real and
# the frontend's server-side Ed25519 verification is exercised end-to-end.
#
#   python make_bundle.py             -> the SHIP bundle (all gates PASS)
#   python make_bundle.py --failing   -> a DON'T_SHIP bundle (toxicity spike +
#                                        expert 4 starved) for the forensic view
#   python make_bundle.py --rogue     -> a FORGED SHIP bundle signed by an
#                                        unpinned key (security demo: the math
#                                        verifies, the issuer is untrusted)
#   python make_bundle.py --restore   -> make the trusted SHIP bundle "latest" again
import os
import sys
from pathlib import Path

from tinct.security.evidence import EvidenceReport
from tinct.security.signing import SigningKey

PROJECT = Path(__file__).parent
KEYS = PROJECT / ".tinct" / "keys"
EVIDENCE = PROJECT / ".tinct" / "evidence"
SHIP_RUN = "cert_20260904_143022"
ROGUE_RUN = "cert_20260905_120000_rogue"


def load_or_create_key() -> SigningKey:
    key_path = KEYS / "default_private.pem"
    if key_path.is_file():
        return SigningKey.load(KEYS, "default")
    key = SigningKey.generate("default")
    key.save(KEYS)
    return key


def load_or_create_rogue_key() -> SigningKey:
    """A separate, deliberately UNPINNED key — the 'attacker' identity."""
    key_path = KEYS / "rogue_private.pem"
    if key_path.is_file():
        return SigningKey.load(KEYS, "rogue")
    key = SigningKey.generate("rogue")
    key.save(KEYS)
    return key


def passing_gates() -> dict:
    return {
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


def failing_gates() -> dict:
    return {
        "result": "FAIL",
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
            "adapter_refusal_rate": 0.81,
            "regression_rate": 0.047,
            "threshold": 0.2,
        },
        "toxicity": {
            "status": "FAIL",
            "method": "heuristic",
            "base_toxicity_avg": 0.02,
            "adapter_toxicity_avg": 0.068,
            "increase_factor": 3.4,
            "threshold": 2.0,
        },
        "expert_collapse": {
            "status": "FAIL",
            "num_experts": 8,
            "total_tokens_routed": 5120,
            "min_utilization": 0.0,
            "max_utilization": 0.226,
            "laziest_expert_id": 4,
            "threshold": 0.01,
        },
        "routing_regression": {
            "status": "FAIL",
            "num_experts": 8,
            "regressed_experts": [4],
            "base_utilization": [0.125, 0.122, 0.13, 0.128, 0.119, 0.124, 0.126, 0.126],
            "adapter_utilization": [0.148, 0.144, 0.146, 0.143, 0.0, 0.146, 0.137, 0.136],
            "adapter_over_base_ratio": [1.184, 1.18, 1.123, 1.117, 0.0, 1.177, 1.087, 1.079],
            "relative_drop_threshold": 0.5,
            "base_floor": 0.02,
        },
        "offload_stats": {
            "h2d_streams": 138,
            "d2h_evictions": 131,
            "cache_hits": 8210,
            "bytes_h2d": 4294967296,
            "bytes_d2h": 4180000000,
        },
    }


def write_bundle(key: SigningKey, run_name: str, gates: dict, decision: str,
                 project: str = "demo-cert") -> Path:
    report = EvidenceReport(
        project_name=project,
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        family="mistral",
        decision=decision,
        artifacts={
            "adapter": {
                "path": f"runs/{run_name}/adapter",
                "dir": True,
                "files": [
                    {"path": f"runs/{run_name}/adapter/adapter_model.safetensors", "sha256": "9f2c02a71b4d1e5f0c8a6d3e7b9c1a4f2e8d6b0c3a5f7e9d1c3b5a7f9e1d3c5b"},
                    {"path": f"runs/{run_name}/adapter/adapter_config.json", "sha256": "1a2b3c4d5e6f70819a2b3c4d5e6f70819a2b3c4d5e6f70819a2b3c4d5e6f7081a"},
                ],
                "adapter_sha256": "4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b",
            },
            "dataset": {"path": "examples/mistral_data.jsonl", "sha256": "7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d", "size": 18432},
        },
        data_report={"format": "instruct", "rows": 16, "errors": 0, "model_family": "mistral"},
        eval_report={"gate": "generation_smoke_test", "status": "PASS", "empty_responses": 0, "repetitive_responses": 0},
        metrics={"final_loss": 1.42, "steps": 24},
        config={"method": "certify", "lora_r": 16, "max_seq_len": 2048},
        safety_gates=gates,
        training_tool="unsloth",
        training_executed=False,
    )
    report.sign(key)
    path = report.write(EVIDENCE, run_name)
    print(f"verified: {report.verify()}  wrote: {path}")
    return path


if __name__ == "__main__":
    if "--restore" in sys.argv:
        good = EVIDENCE / f"{SHIP_RUN}_evidence.json"
        if good.is_file():
            os.utime(good)  # newest mtime -> selected as "latest"
            print(f"Restored: {good.name} is newest — the dashboard shows the trusted run.")
        else:
            print("No SHIP bundle found; run `python make_bundle.py` first.")
        sys.exit(0)

    if "--rogue" in sys.argv:
        rogue = load_or_create_rogue_key()
        write_bundle(rogue, ROGUE_RUN, passing_gates(), "SHIP", project="rogue-cert")
        print()
        print("  ROGUE BUNDLE — security demo only.")
        print("  Signed by an UNPINNED key with a FORGED 'SHIP' verdict: the Ed25519")
        print("  math verifies, but the dashboard must show 'UNTRUSTED ISSUER' and")
        print("  refuse the download (403).")
        print("  Back to the trusted run:  python demo-cert/make_bundle.py --restore")
        sys.exit(0)

    key = load_or_create_key()
    if "--failing" in sys.argv:
        write_bundle(key, "cert_20260904_160000_failing", failing_gates(), "DON'T_SHIP")
    else:
        write_bundle(key, SHIP_RUN, passing_gates(), "SHIP")
