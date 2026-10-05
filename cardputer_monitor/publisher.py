from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests


class PublishError(RuntimeError):
    """A public error deliberately excluding credentials and response bodies."""


class TelemetryPublisher:
    def __init__(self, url: str, token_file: Path, *, timeout: float = 10, session=None, sleep=time.sleep):
        parsed = urlsplit(url)
        if (any(char.isspace() or ord(char) < 32 for char in url)
                or parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("telemetry endpoint must use HTTPS without URL credentials")
        self.url, self.token_file, self.timeout = url, Path(token_file), timeout
        self.session = session if session is not None else requests.Session()
        # Do not discover unrelated credentials from .netrc or ambient proxies.
        self.session.trust_env = False
        self.sleep = sleep

    def _token(self) -> str:
        try:
            with self.token_file.open("r", encoding="utf-8") as source:
                value = source.read(513).strip()
        except (OSError, UnicodeError):
            raise PublishError("telemetry token file is unavailable") from None
        if not value or len(value) > 512 or any(ord(char) < 33 or ord(char) > 126 for char in value):
            raise PublishError("telemetry token file is invalid")
        return value

    def publish(self, snapshot: dict) -> None:
        token = self._token()
        # Validate strict JSON before network I/O; retries use the same sample.
        body = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(body) > 64 * 1024:
            raise PublishError("telemetry payload exceeds 64 KiB")
        failure = "telemetry publish failed"
        for attempt in range(2):
            try:
                response = self.session.post(
                    self.url, data=body,
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    timeout=self.timeout, allow_redirects=False, verify=True,
                )
            except requests.RequestException as exc:
                failure = f"telemetry connection failed: {type(exc).__name__}"
            else:
                code = response.status_code
                response.close()
                if 200 <= code < 300:
                    return
                failure = f"telemetry HTTP {code}"
                # Never retry authentication, validation or redirect failures.
                if code < 500:
                    raise PublishError(failure)
            if attempt == 0:
                self.sleep(1)
        raise PublishError(failure)
