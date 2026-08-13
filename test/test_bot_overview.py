import json
import tempfile
import time
from pathlib import Path

from macro_control.bot_overview import BotOverviewProvider


class FakeAPI:
    def all_bot_statuses(self):
        return {
            "status": "success",
            "data": {
                "grid-live-fdusd-400": {
                    "status": "running",
                    "source": "docker",
                    "recently_active": True,
                    "performance": {},
                    "error_logs": [],
                    "general_logs": [{"msg": "must not leak"}],
                },
                "dca-live-btcusdt-200": {
                    "status": "running",
                    "source": "mqtt",
                    "recently_active": True,
                    "performance": {
                        "btc": {
                            "status": "running",
                            "performance": {
                                "global_pnl_quote": 1.25,
                                "global_pnl_pct": 0.5,
                                "positions_summary": ["not exposed"],
                            },
                        }
                    },
                    "error_logs": [{"msg": "one"}],
                },
            },
        }


def test_bot_overview_lists_plain_scripts_and_grid_mtm():
    now = time.time()
    state = {
        "armed": True,
        "shadow": False,
        "emergency_ready": True,
        "xgboost_risk_gate": {
            "schema": "ethbtc-forced-exit-live-contract-v1",
            "pairs": {"BTC-FDUSD": {"buy_enabled": False}},
        },
        "bots": {
            "grid-live-fdusd-400": {
                "latest": {
                    "pnl": "0.42",
                    "equity": "420.42",
                    "peak_equity": "421",
                    "drawdown_pct": "0.0013",
                    "observed_at": now,
                    "pairs": {"BTC-FDUSD": {"pnl": "0.20"}},
                }
            }
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "guard_state.json"
        path.write_text(json.dumps(state), encoding="utf-8")
        result = BotOverviewProvider(FakeAPI(), path).snapshot()

    assert result["running_count"] == 2
    assert {bot["bot_name"] for bot in result["bots"]} == {
        "dca-live-btcusdt-200",
        "grid-live-fdusd-400",
    }
    grid_bot = next(
        bot for bot in result["bots"] if bot["bot_name"] == "grid-live-fdusd-400"
    )
    assert grid_bot["controllers"] == {}
    assert result["grid"]["mtm_pnl_quote"] == "0.42"
    assert result["grid"]["pnl_method"] == "strategy_owned_mark_to_market"
    assert result["grid"]["fresh"] is True
    assert result["grid"]["technical_buy_gate"]["pairs"]["BTC-FDUSD"]["buy_enabled"] is False
    assert "general_logs" not in json.dumps(result)
    dca = next(bot for bot in result["bots"] if bot["bot_name"].startswith("dca-"))
    assert dca["error_count"] == 1
    assert dca["controllers"]["btc"]["performance"] == {
        "global_pnl_quote": 1.25,
        "global_pnl_pct": 0.5,
    }
