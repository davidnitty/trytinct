"""Model prep — chunk a safetensors model for low-VRAM streaming.

Inspired by AirLLM. A massive safetensors model is copied into
streamable chunks under the local ``.tinct`` cache, and every chunk is
SHA-256-hashed so the manifest can be folded into the ship evidence bundle.

Security rule (fail-closed): only safetensors models are accepted. Pickle
``.bin`` checkpoints are blocked outright.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from tinct.utils.logging import get_logger

log = get_logger("tinct.engine.chunking")


class ModelChunker:
    """Splits a safetensors model into streamable local chunks.

    Usage::

        chunker = ModelChunker(cache_dir)
        manifest = chunker.chunk_model(model_path, chunk_size_mb=500)
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir / "chunks"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def chunk_model(self, model_path: Path, chunk_size_mb: int = 500) -> dict:
        """Chunk the model and return a manifest of chunk hashes.

        Args:
            model_path: Directory containing ``model.safetensors.index.json``
                and the sharded ``*.safetensors`` weight files.
            chunk_size_mb: Target chunk size. The current grouping logic
                streams whole shard files (tensor-level grouping by byte size
                is noted as production follow-up below).

        Raises:
            ValueError: If the model lacks a safetensors index (e.g. pickle
                ``.bin`` weights are present instead) — fail-closed.
        """
        # Fail-closed: safetensors only, no pickle. A sharded model provides an
        # index; a single-file safetensors checkpoint is also accepted.
        index_file = model_path / "model.safetensors.index.json"
        if not index_file.exists() and not any(model_path.glob("*.safetensors")):
            raise ValueError(
                "tinct requires safetensors weights (a model.safetensors file "
                "or a model.safetensors.index.json). Pickle .bin files are blocked."
            )

        weight_map = {}
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as fh:
                index_data = json.load(fh)
            # The index maps tensor names -> shard file.
            weight_map = index_data["weight_map"]
            # Production follow-up: group tensors by measured byte size so a
            # chunk stays within ``chunk_size_mb`` instead of whole shard files.
            del weight_map

        chunks_manifest: dict = {}
        log.info("Chunking model for low-VRAM streaming: %s", model_path)

        for safetensor_file in model_path.glob("*.safetensors"):
            dest_path = self.cache_dir / safetensor_file.name
            if not dest_path.exists():
                shutil.copy2(safetensor_file, dest_path)

            chunk_hash = self._hash_file(dest_path)
            chunks_manifest[safetensor_file.name] = chunk_hash
            log.debug("chunked %s -> %s", safetensor_file.name, chunk_hash)

        return chunks_manifest

    @staticmethod
    def _hash_file(filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as fh:
            for byte_block in iter(lambda: fh.read(4096), b""):
                sha256.update(byte_block)
        return sha256.hexdigest()