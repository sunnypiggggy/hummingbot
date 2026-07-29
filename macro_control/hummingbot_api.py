from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass


class HummingbotAPIError(RuntimeError):
    pass


@dataclass
class HummingbotAPI:
    base_url: str
    username: str
    password: str
    timeout: float = 10.0

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode()
        token = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        request = urllib.request.Request(
            self.base_url.rstrip("/") + path,
            data=body,
            method=method,
            headers={
                "Authorization": f"Basic {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise HummingbotAPIError(f"{method} {path} failed: {exc}") from exc
        if not raw:
            return {}
        value = json.loads(raw.decode())
        return value if isinstance(value, dict) else {"items": value}

    def bot_status(self, bot_name: str) -> dict:
        return self._request("GET", f"/bot-runs/{bot_name}")

    def update_controller(
        self, bot_name: str, controller_name: str, profile: dict
    ) -> dict:
        return self._request(
            "POST",
            f"/controllers/bots/{bot_name}/{controller_name}/config",
            profile,
        )
