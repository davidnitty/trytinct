"""
MoE Safety Gates: Expert Collapse Detection.

Detects if a Mixture-of-Experts (MoE) model's router becomes lazy during
fine-tuning and stops utilizing the full expert pool.

Works by attaching PyTorch forward hooks to the router (gate) layers
and tallying expert selections during generation.
"""

import logging
from typing import Callable

from tinct.engine.moe import iter_moe_routers

log = logging.getLogger(__name__)


class MoEExpertTracker:
    """Tracks expert utilization across all MoE layers in a model."""

    def __init__(self, num_experts: int, top_k: int = 2):
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_counts = [0] * num_experts
        self.total_selections = 0
        self.hooks = []

    def _hook_fn(self, module, input_args, output):
        """
        PyTorch forward hook for the router 'gate' layer.
        output = router_logits of shape [batch_size * seq_len, num_experts]
        """
        import torch

        router_logits = output

        # Find the top-k experts for each token
        # Mixtral uses top-2 by default
        top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1).indices

        # Tally the selections
        flat_indices = top_k_indices.flatten().tolist()
        for idx in flat_indices:
            if 0 <= idx < self.num_experts:
                self.expert_counts[idx] += 1
                self.total_selections += 1

    def attach_to_model(self, model):
        """
        Scans the model for MoE router layers and attaches hooks.
        Works for Mixtral, Qwen-MoE, DeepSeek, and similar architectures
        (structural detection — see :func:`tinct.engine.moe.iter_moe_routers`).
        """
        attached_count = 0
        for _name, block in iter_moe_routers(model):
            handle = block.gate.register_forward_hook(self._hook_fn)
            self.hooks.append(handle)
            attached_count += 1

        if attached_count > 0:
            log.info("[tinct] Attached MoE tracker to %d router layers.", attached_count)
        else:
            log.warning("[tinct] No MoE router layers found to track.")
        return attached_count

    def detach(self):
        """Removes all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def reset(self) -> None:
        """Zero tallies between measurement passes."""
        self.expert_counts = [0] * self.num_experts
        self.total_selections = 0

    def check_collapse(self, min_utilization_threshold: float = 0.01) -> dict:
        """
        Checks if any expert is being severely underutilized.

        Args:
            min_utilization_threshold: Minimum % of traffic an expert must receive.
                                       Default 0.01 (1%). Mixtral has 8 experts,
                                       so uniform routing would be 12.5% each.
        """
        if self.total_selections == 0:
            return {
                "status": "NOT_CONFIGURED",
                "reason": "No tokens were routed through MoE layers.",
            }

        utilizations = [count / self.total_selections for count in self.expert_counts]
        min_util = min(utilizations)
        max_util = max(utilizations)

        # Find the laziest expert
        laziest_expert_id = utilizations.index(min_util)

        status = "PASS"
        if min_util < min_utilization_threshold:
            status = "FAIL"
            log.warning(
                "[tinct] Expert Collapse Detected! Expert %d received only %.4f%% of traffic.",
                laziest_expert_id,
                min_util * 100,
            )

        return {
            "status": status,
            "num_experts": self.num_experts,
            "total_tokens_routed": self.total_selections // self.top_k,
            "min_utilization": round(min_util, 4),
            "max_utilization": round(max_util, 4),
            "laziest_expert_id": laziest_expert_id,
            "threshold": min_utilization_threshold,
        }


def check_expert_collapse(
    model_callable: Callable,
    prompts: list,
    num_experts: int = 8,
    top_k: int = 2,
    min_utilization: float = 0.01,
    model_instance=None,  # We need the raw model to attach hooks
) -> dict:
    """
    Generates responses and checks for expert collapse.

    Note: In production, `model_instance` must be passed so we can attach hooks.
    If not passed, we cannot track MoE routing.
    """
    if model_instance is None:
        return {
            "status": "NOT_CONFIGURED",
            "reason": "Raw model instance not provided to MoE gate.",
        }

    tracker = MoEExpertTracker(num_experts=num_experts, top_k=top_k)
    tracker.attach_to_model(model_instance)

    try:
        # Generate responses to trigger the hooks
        for prompt in prompts:
            _ = model_callable(prompt)

        result = tracker.check_collapse(min_utilization_threshold=min_utilization)

    finally:
        # ALWAYS clean up hooks, even if generation crashes
        tracker.detach()

    return result


def check_routing_regression(
    model_callable: Callable,          # adapter pass
    base_model_callable: Callable,     # base pass (disable_adapter)
    prompts: list,
    num_experts: int,
    top_k: int = 2,
    relative_drop_threshold: float = 0.5,  # expert loses >50% of its base share
    base_floor: float = 0.02,              # only judge experts base used >=2%
    model_instance=None,
) -> dict:
    """
    Flags experts the adapter starved relative to the BASE model.

    An expert is 'regressed' only if:
      - base utilization >= base_floor (base actually used it), AND
      - adapter utilization < base utilization * (1 - relative_drop_threshold)

    Experts the base already starved are ignored — the adapter isn't blamed
    for the base model's laziness.
    """
    if model_instance is None:
        return {"status": "NOT_CONFIGURED",
                "reason": "Raw model instance not provided to MoE gate."}

    tracker = MoEExpertTracker(num_experts=num_experts, top_k=top_k)
    tracker.attach_to_model(model_instance)

    try:
        for prompt in prompts:
            _ = model_callable(prompt)          # adapter pass
        adapter_counts = list(tracker.expert_counts)
        adapter_total = tracker.total_selections
        tracker.reset()

        for prompt in prompts:
            _ = base_model_callable(prompt)     # base pass
        base_counts = list(tracker.expert_counts)
        base_total = tracker.total_selections
    finally:
        tracker.detach()

    if adapter_total == 0 or base_total == 0:
        return {"status": "NOT_CONFIGURED",
                "reason": "No tokens routed during regression measurement."}

    u_base = [c / base_total for c in base_counts]
    u_adp = [c / adapter_total for c in adapter_counts]

    regressed = [
        i for i in range(num_experts)
        if u_base[i] >= base_floor
        and u_adp[i] < u_base[i] * (1 - relative_drop_threshold)
    ]

    ratios = [
        round(u_adp[i] / u_base[i], 4) if u_base[i] > 0 else None
        for i in range(num_experts)
    ]

    return {
        "status": "FAIL" if regressed else "PASS",
        "num_experts": num_experts,
        "regressed_experts": regressed,
        "base_utilization": [round(u, 4) for u in u_base],
        "adapter_utilization": [round(u, 4) for u in u_adp],
        "adapter_over_base_ratio": ratios,
        "relative_drop_threshold": relative_drop_threshold,
        "base_floor": base_floor,
    }
