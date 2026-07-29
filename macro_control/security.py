from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Mapping


def canonical_request(
    method: str, path: str, timestamp: str, nonce: str, body: bytes
) -> bytes:
    digest = hashlib.sha256(body).hexdigest()
    return "\n".join((method.upper(), path, timestamp, nonce, digest)).encode()


def sign_request(
    secret: str, method: str, path: str, timestamp: str, nonce: str, body: bytes
) -> str:
    return hmac.new(
        secret.encode(),
        canonical_request(method, path, timestamp, nonce, body),
        hashlib.sha256,
    ).hexdigest()


@dataclass
class NonceCache:
    ttl_seconds: int = 120
    state_path: Path | None = None
    _seen: dict[str, float] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def claim(self, nonce: str, now: float | None = None) -> bool:
        with self._lock:
            now = time.time() if now is None else now
            if self.state_path and self.state_path.exists():
                try:
                    persisted = json.loads(self.state_path.read_text(encoding="utf-8"))
                    self._seen.update(
                        {str(key): float(value) for key, value in persisted.items()}
                    )
                except (OSError, ValueError, TypeError):
                    return False
            self._seen = {
                key: expires for key, expires in self._seen.items() if expires > now
            }
            if not nonce or nonce in self._seen:
                return False
            self._seen[nonce] = now + self.ttl_seconds
            if self.state_path:
                self._persist()
            return True

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            dir=self.state_path.parent, prefix=".nonces-", suffix=".tmp", text=True
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(self._seen, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.state_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def verify_request(
    secret: str | Mapping[str, str],
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    body: bytes,
    signature: str,
    nonces: NonceCache,
    key_id: str = "",
    now: float | None = None,
    max_skew_seconds: int = 60,
    max_body_bytes: int = 64 * 1024,
) -> tuple[bool, str]:
    now = time.time() if now is None else now
    if len(body) > max_body_bytes:
        return False, "body_too_large"
    try:
        request_time = float(timestamp)
    except ValueError:
        return False, "invalid_timestamp"
    if abs(now - request_time) > max_skew_seconds:
        return False, "stale_timestamp"
    if isinstance(secret, Mapping):
        selected_secret = secret.get(key_id)
        if not key_id or not selected_secret:
            return False, "unknown_key_id"
    else:
        selected_secret = secret
    expected = sign_request(selected_secret, method, path, timestamp, nonce, body)
    if not hmac.compare_digest(expected, signature):
        return False, "invalid_signature"
    if not nonces.claim(nonce, now):
        return False, "replayed_nonce"
    return True, "ok"
