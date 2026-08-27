"""tinct — CLI-first post-training stack for LLMs.

tinct validates instruction data, fine-tunes a Llama adapter (LoRA/QLoRA),
evaluates the result, and produces a SHIP / DON'T-SHIP decision backed by a
signed cryptographic evidence report.

The core package deliberately imports only lightweight dependencies at
import-time. Heavy ML packages (torch, transformers, TRL, ...) are optional
extras loaded lazily by the ``engine`` subpackage when a command needs them.
"""

__version__ = "0.5.0"

__all__ = ["__version__"]
