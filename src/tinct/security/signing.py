"""Ed25519 signing keys for ship evidence.

Uses the ``cryptography`` package (lightweight, already a core dependency).
Keys are stored as PEM files under ``.tinct/keys/<name>_*.pem`` with restrictive
permissions on POSIX. The private key never leaves the local project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

PRIVATE_SUFFIX = "_private.pem"
PUBLIC_SUFFIX = "_public.pem"


@dataclass
class SigningKey:
    """A loaded Ed25519 keypair."""

    name: str
    private: Ed25519PrivateKey
    public: Ed25519PublicKey

    @classmethod
    def generate(cls, name: str) -> "SigningKey":
        private = Ed25519PrivateKey.generate()
        return cls(name=name, private=private, public=private.public_key())

    # -- persistence --------------------------------------------------------

    def save(self, keys_dir: Path) -> tuple[Path, Path]:
        keys_dir.mkdir(parents=True, exist_ok=True)
        private_path = keys_dir / f"{self.name}{PRIVATE_SUFFIX}"
        public_path = keys_dir / f"{self.name}{PUBLIC_SUFFIX}"

        private_path.write_bytes(self._private_pem())
        public_path.write_bytes(self._public_pem())

        try:
            private_path.chmod(0o600)  # owner read/write only
            public_path.chmod(0o644)
        except OSError:  # pragma: no cover - Windows may not support chmod
            pass
        return private_path, public_path

    @classmethod
    def load(cls, keys_dir: Path, name: str) -> "SigningKey":
        private_path = keys_dir / f"{name}{PRIVATE_SUFFIX}"
        if not private_path.is_file():
            raise FileNotFoundError(
                f"No signing key {name!r} found in {keys_dir}. "
                "Run `tinct security key generate` first."
            )
        private = serialization.load_pem_private_key(
            private_path.read_bytes(), password=None
        )
        if not isinstance(private, Ed25519PrivateKey):
            raise ValueError(f"Key {name!r} is not an Ed25519 private key.")
        return cls(name=name, private=private, public=private.public_key())

    # -- crypto -------------------------------------------------------------

    def sign_bytes(self, data: bytes) -> bytes:
        return self.private.sign(data)

    def public_pem(self) -> bytes:
        return self._public_pem()

    def _private_pem(self) -> bytes:
        return self.private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def _public_pem(self) -> bytes:
        return self.public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )


def public_key_from_pem(pem: bytes) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(
        serialization.load_pem_public_key(pem).public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )


def verify_signature(public_pem: bytes, data: bytes, signature: bytes) -> bool:
    """Return True if ``signature`` verifies over ``data`` with ``public_pem``."""
    try:
        key = public_key_from_pem(public_pem)
        key.verify(signature, data)
        return True
    except Exception:
        return False
