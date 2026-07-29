from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock

from .telemetry import build_sanitized_snapshot


class JsonFileTelemetryProvider:
    """Adapter for an OCI-local collector; the input file is never exposed raw."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: dict[str, dict] = {}
        self._lock = Lock()

    def snapshot(self) -> dict:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        observed_at = (
            datetime.fromisoformat(str(payload["observed_at"]))
            if payload.get("observed_at")
            else None
        )
        snapshot = build_sanitized_snapshot(
            payload.get("bot_statuses", []),
            payload.get("market", {}),
            observed_at=observed_at,
        )
        with self._lock:
            self._cache[snapshot["snapshot_id"]] = snapshot
            if len(self._cache) > 32:
                for key in list(self._cache)[:-32]:
                    del self._cache[key]
        return snapshot

    def snapshot_by_id(self, snapshot_id: str) -> dict | None:
        with self._lock:
            return self._cache.get(snapshot_id)
