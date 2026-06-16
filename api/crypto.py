"""
AES-256-GCM Encryption / Decryption
All API payloads are encrypted at the application layer (on top of TLS).

Key derivation: PBKDF2-HMAC-SHA256 from shared secret in .env
Cipher: AES-256-GCM (authenticated encryption — provides both confidentiality + integrity)
Nonce: 12 bytes, cryptographically random, prepended to ciphertext

Wire format (base64-encoded):
    [12-byte nonce] + [ciphertext] + [16-byte GCM auth tag]
"""

import os
import base64
import json
import hashlib
from typing import Any, Union

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend


# ---------------------------------------------------------------------------
# Key Derivation
# ---------------------------------------------------------------------------

# Salt is fixed and known to both parties (not secret — just ensures
# the derived key is domain-specific). Change if you fork this project.
_KDF_SALT = b"icu-biosignal-pipeline-v1-salt-2026"
_KEY_ITERATIONS = 260_000   # NIST recommended minimum for PBKDF2-SHA256


def derive_key(shared_secret: str) -> bytes:
    """
    Derive a 256-bit AES key from the shared secret string.
    Uses PBKDF2-HMAC-SHA256 with a fixed domain salt.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,       # 256 bits
        salt=_KDF_SALT,
        iterations=_KEY_ITERATIONS,
        backend=default_backend(),
    )
    return kdf.derive(shared_secret.encode("utf-8"))


# ---------------------------------------------------------------------------
# Load key from environment (cached)
# ---------------------------------------------------------------------------

_cached_key: bytes | None = None


def get_aes_key() -> bytes:
    """Get the derived AES-256 key, loading from env on first call."""
    global _cached_key
    if _cached_key is None:
        secret = os.environ.get("AES_SHARED_SECRET", "")
        if not secret:
            raise RuntimeError(
                "AES_SHARED_SECRET is not set in environment. "
                "Copy .env.example to .env and set a strong secret."
            )
        _cached_key = derive_key(secret)
    return _cached_key


# ---------------------------------------------------------------------------
# Encryption / Decryption
# ---------------------------------------------------------------------------

def encrypt_payload(data: Any) -> str:
    """
    Serialize data to JSON, encrypt with AES-256-GCM.
    Returns base64-encoded string: nonce(12) + ciphertext + tag(16)
    """
    key = get_aes_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    plaintext = json.dumps(data, default=str).encode("utf-8")
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)  # AAD = None
    # ciphertext already includes the 16-byte GCM tag appended by cryptography lib
    combined = nonce + ciphertext
    return base64.b64encode(combined).decode("ascii")


def decrypt_payload(encrypted_b64: str) -> Any:
    """
    Decrypt a base64-encoded AES-256-GCM payload.
    Returns the original Python object (parsed from JSON).
    Raises ValueError on authentication failure or bad format.
    """
    key = get_aes_key()
    aesgcm = AESGCM(key)
    try:
        combined = base64.b64decode(encrypted_b64)
        nonce = combined[:12]
        ciphertext = combined[12:]
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(plaintext.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Decryption failed: {exc}") from exc


# ---------------------------------------------------------------------------
# FastAPI response wrapper
# ---------------------------------------------------------------------------

def encrypted_response(data: Any) -> dict:
    """
    Wrap data in a standard encrypted API response envelope.
    The client must base64-decode then AES-GCM decrypt the 'payload' field.
    """
    return {
        "encrypted": True,
        "algorithm": "AES-256-GCM",
        "payload": encrypt_payload(data),
    }
