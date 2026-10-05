from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

TELEMETRY_URL = "https://sh.sunnypiggy.top/api/cardputer/v1/telemetry/trading"


@dataclass(frozen=True)
class Config:
    reports_dir: Path = Path("/reports")
    token_file: Path = Path("/run/secrets/cardputer_telemetry_token")
    telemetry_url: str = TELEMETRY_URL
    interval_seconds: int = 60
    request_timeout_seconds: float = 10
    price_provider: str = "none"
    price_timeout_seconds: float = 5

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {"reports_dir", "token_file", "telemetry_url", "interval_seconds", "request_timeout_seconds", "price_provider", "price_timeout_seconds"}
        if not isinstance(data, dict) or set(data) - allowed:
            raise ValueError("unknown configuration fields")
        for key in ("reports_dir", "token_file", "telemetry_url"):
            if key in data and (not isinstance(data[key], str) or not data[key]):
                raise ValueError(f"invalid {key}")
        if "interval_seconds" in data and (type(data["interval_seconds"]) is not int or data["interval_seconds"] < 60):
            raise ValueError("interval_seconds must be at least 60")
        timeout = data.get("request_timeout_seconds", 10)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 20:
            raise ValueError("request_timeout_seconds must be between 1 and 20")
        if data.get("price_provider", "none") not in ("none", "binance-public"):
            raise ValueError("price_provider must be none or binance-public")
        price_timeout = data.get("price_timeout_seconds", 5)
        if isinstance(price_timeout, bool) or not isinstance(price_timeout, (int, float)) or not 1 <= price_timeout <= 10:
            raise ValueError("price_timeout_seconds must be between 1 and 10")
        url = data.get("telemetry_url", TELEMETRY_URL)
        parsed = urlsplit(url)
        if (any(char.isspace() or ord(char) < 32 for char in url)
                or parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path != "/api/cardputer/v1/telemetry/trading"):
            raise ValueError("telemetry_url must be the HTTPS trading telemetry endpoint")
        for key in ("reports_dir", "token_file"):
            if key in data:
                data[key] = Path(data[key])
        return cls(**data)
