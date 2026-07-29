from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import Lock


class StateStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "state.json"
        self.audit_path = self.directory / "audit.jsonl"
        self._lock = Lock()

    def read(self) -> dict:
        if not self.state_path.exists():
            return self._empty_state()
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 3:
            raise ValueError("unsupported macro state schema; only version 3 is accepted")
        value.setdefault("decisions", {})
        value.setdefault("leases", {})
        value.setdefault("bot_gate_state", {})
        value.setdefault("desired_gates", {"buy": True, "sell": True})
        value.setdefault("last_reconcile", None)
        value.setdefault("retry_state", {})
        value.setdefault("approval_callbacks", {})
        return value

    @staticmethod
    def _empty_state() -> dict:
        return {
            "schema_version": 3,
            "decisions": {},
            "leases": {},
            "bot_gate_state": {},
            "desired_gates": {"buy": True, "sell": True},
            "last_reconcile": None,
            "retry_state": {},
            "approval_callbacks": {},
        }

    def write(self, value: dict) -> None:
        payload = json.dumps(value, indent=2, sort_keys=True)
        with self._lock:
            handle, temp_name = tempfile.mkstemp(
                dir=self.directory, prefix=".state-", suffix=".tmp", text=True
            )
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_name, self.state_path)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)

    def append_audit(self, event: dict) -> None:
        line = json.dumps(event, sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())
