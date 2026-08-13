"""Streaming inference engine — AirLLM-style layer streaming for tinct.

Wraps a Hugging Face causal LM during ``tinct eval`` (or the forward pass of
``tinct train``): the architecture is created on the ``meta`` device so it
allocates 0 bytes of RAM, then transformer blocks can be streamed one-by-one
into VRAM from the local chunk cache produced by
:mod:`tinct.engine.chunking`.

Heavy imports (torch/transformers/accelerate) happen lazily inside methods so
this module — like the rest of the core — never forces ML dependencies at
import time.
"""

from __future__ import annotations

from pathlib import Path

from tinct.engine.deps import ensure_train_deps
from tinct.utils.logging import get_logger

log = get_logger("tinct.engine.streaming")


class StreamingInferenceEngine:
    """AirLLM-style streaming forward pass for tinct evals.

    Usage::

        engine = StreamingInferenceEngine(model_path=..., chunk_dir=...)
        try:
            engine.load_meta_model()
            hidden = engine.stream_forward(input_ids)
        finally:
            engine.cleanup()          # fail-closed: always free VRAM
    """

    def __init__(self, model_path: Path, chunk_dir: Path) -> None:
        self.model_path = Path(model_path)
        self.chunk_dir = Path(chunk_dir)
        self.device = "cpu"  # resolved to cuda in load_meta_model when available

    # -- lifecycle ----------------------------------------------------------

    def load_meta_model(self) -> None:
        """Initialize the architecture without allocating RAM.

        The magic trick: weights are constructed under
        ``accelerate.init_empty_weights`` so the model lives on the ``meta``
        device and consumes 0 bytes of RAM.
        """
        ensure_train_deps()
        import torch  # noqa: F401 - used indirectly for dtype/device checks
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM

        log.info("[tinct streaming] Initializing meta-device architecture...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = AutoConfig.from_pretrained(self.model_path)

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(
                config, torch_dtype=torch.float16
            )
        # Keep the meta placement explicit and safe against accidental RAM use.
        self.model = self.model.to("meta")
        log.info("[tinct streaming] Architecture on %r; weights will stream from %s",
                 "meta", self.chunk_dir)

    def stream_forward(self, input_ids: "torch.Tensor"):
        """Execute a forward pass by loading one transformer block at a time.

        Scaffold: the real implementation hooks the model's forward pass,
        intercepting each block, loading its weights from ``self.chunk_dir``
        via safetensors, moving to ``self.device``, computing, and evicting.
        That integration lands in the next engine stage — until then this
        fails closed instead of silently returning ``None``.
        """
        raise NotImplementedError(
            "stream_forward is scaffolded; layer-by-layer streaming hooks are "
            "implemented in the next engine stage."
        )

    def cleanup(self) -> None:
        """Ensure VRAM is freed. Fail-closed: call in a ``finally`` block."""
        if hasattr(self, "model"):
            try:
                del self.model
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            import torch
        except ImportError:  # torch not installed; nothing to free
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.debug("[tinct streaming] cleanup complete")