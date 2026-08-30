"""
MoE Safety Gates: Expert Collapse Detection.

Detects if a Mixture-of-Experts (MoE) model's router becomes lazy during
fine-tuning and stops utilizing the full expert pool.

Works by attaching PyTorch forward hooks to the router (gate) layers
and tallying expert selections during generation.
"""

import logging
from typing import Callable, Iterator, Tuple

log = logging.getLogger(__name__)

# Mixtral names its MoE block "block_sparse_moe" and its router "gate";
# the router itself is an nn.Linear whose out_features == num_experts.
_ROUTER_MARKER = "block_sparse_moe"


def iter_moe_routers(model) -> Iterator[Tuple[str, object]]:
    """Yield ``(name, gate_module)`` for every MoE router in ``model``.

    Works for Mixtral-style architectures (``block_sparse_moe.gate``) and is
    the single scan used both by :class:`MoEExpertTracker` and the pipeline
    integration in :mod:`tinct.safety.gates`, so gate applicability and hook
    attachment can never disagree.
    """
    for name, module in model.named_modules():
        if _ROUTER_MARKER in name and hasattr(module, "gate"):
            yield name, module.gate


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
        Works for Mixtral, Qwen-MoE, and similar architectures.
        """
        attached_count = 0
        for _name, gate in iter_moe_routers(model):
            handle = gate.register_forward_hook(self._hook_fn)
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
