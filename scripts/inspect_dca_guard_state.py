#!/usr/bin/env python3
"""Print a non-sensitive summary of the DCA Guard state."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    state_dir = Path(os.getenv("DCA_LIVE_STATE_PATH", "/workspace/state"))
    state = json.loads((state_dir / "guard_state.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "last_success_at": state.get("last_success_at"),
        "last_monitor_error": state.get("last_monitor_error"),
        "bots": {
            name: {
                "tripped": value.get("tripped"),
                "action_complete": value.get("action_complete"),
                "trip_reason": value.get("trip_reason"),
                "started_at": value.get("started_at"),
                "latest_updated_at": (value.get("latest") or {}).get("updated_at"),
            }
            for name, value in state.get("bots", {}).items()
            if name.startswith("dca-live-")
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
