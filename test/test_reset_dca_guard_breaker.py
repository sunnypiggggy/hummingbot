from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.reset_dca_guard_breaker import (
    BOT_NAMES,
    CONFIRMATION,
    EXPECTED_REASON,
    main,
)


def test_reset_routes_flattened_bots_through_guard_reentry(tmp_path: Path) -> None:
    state_path = tmp_path / "guard_state.json"
    state_path.write_text(json.dumps({
        "bots": {
            name: {
                "tripped": True,
                "trip_reason": EXPECTED_REASON,
                "tripped_at": 100.0,
                "action_complete": True,
                "managed_base_target": "0.01",
                "recovery": {
                    "phase": "LATCHED",
                    "remaining_base": {"PAIR": "0"},
                    "exit_completed_at": 110.0,
                },
            }
            for name in BOT_NAMES
        }
    }), encoding="utf-8")

    with (
        patch.dict("os.environ", {"DCA_LIVE_STATE_PATH": str(tmp_path)}),
        patch.object(sys, "argv", ["reset", "--confirm", CONFIRMATION]),
    ):
        assert main() == 0

    state = json.loads(state_path.read_text(encoding="utf-8"))
    for name in BOT_NAMES:
        bot = state["bots"][name]
        assert bot["tripped"] is False
        assert bot["managed_base_target"] == "0.01"
        assert bot["recovery"]["phase"] == "COOLDOWN"
        assert bot["recovery"]["latch_after_exit"] is False
        assert bot["recovery"]["remaining_base"] == {"PAIR": "0"}
