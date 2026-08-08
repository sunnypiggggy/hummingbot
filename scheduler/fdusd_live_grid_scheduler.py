"""Weekly FDUSD live-grid parameter selector.

This service never deploys or arms a bot. It updates a versioned parameter file
that the already-running strategy can apply after cancelling its own orders.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from fdusd_live_grid_optimizer import select_candidate, weekly_cutoff
from grid_live_common import (
    ACTIVE_SELECTION_SCHEMA_VERSION,
    PORTFOLIOS,
    build_live_config,
    effective_take_profit,
)
from grid_macro_gate import build_grid_macro_gate
from ethbtc_forced_exit_contract import MODEL_VERSION as XGBOOST_MODEL_VERSION
from ethbtc_forced_exit_contract import SCHEMA as XGBOOST_GATE_SCHEMA
from validate_grid_live import INTERVAL_SECONDS, load_candles


LOG = logging.getLogger("grid-live-fdusd-scheduler")
SHANGHAI = ZoneInfo("Asia/Shanghai")


class Scheduler:
    def __init__(self) -> None:
        self.root = Path(os.getenv("GRID_LIVE_FDUSD_STATE_PATH", "/workspace/state"))
        self.bots = Path(os.getenv("BOTS_PATH", "/workspace/bots"))
        self.cache = self.root / "candles"
        self.selections = self.root / "selections"
        self.state_path = self.root / "scheduler_state.json"
        self.fee_state_path = self.root / "private_preflight.json"
        self.canonical_selection = self.root / "active_selection.json"
        self.canonical_config = self.root / PORTFOLIOS["FDUSD"].config_name
        self.macro_source = Path(
            os.getenv("GRID_LIVE_MACRO_STATE_PATH", "/workspace/macro/state.json")
        )
        self.canonical_macro_gate = self.root / "macro_gate.json"
        self.macro_max_age_seconds = int(
            os.getenv("GRID_LIVE_MACRO_MAX_AGE_SECONDS", "150")
        )
        self.macro_execution_enabled = (
            os.getenv("GRID_LIVE_FOMC_EXECUTION_ENABLED", "false").lower()
            == "true"
        )
        self.parameter_updates_enabled = (
            os.getenv("GRID_LIVE_PARAMETER_UPDATES_ENABLED", "false").lower()
            == "true"
        )
        self.reconcile_seconds = int(
            os.getenv("GRID_LIVE_GATE_RECONCILE_SECONDS", "5")
        )
        if not 1 <= self.reconcile_seconds <= 10:
            raise ValueError(
                "GRID_LIVE_GATE_RECONCILE_SECONDS must be between 1 and 10."
            )
        if not 30 <= self.macro_max_age_seconds <= 180:
            raise ValueError(
                "GRID_LIVE_MACRO_MAX_AGE_SECONDS must be between 30 and 180."
            )
        self._last_macro_key: tuple | None = None
        for directory in (self.root, self.cache, self.selections):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_json(path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    @staticmethod
    def atomic_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def atomic_yaml(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        temporary.replace(path)

    def run_forever(self) -> None:
        while True:
            try:
                self.reconcile()
            except Exception:
                LOG.exception("FDUSD selection failed; the previous parameter version remains active.")
            time.sleep(self.reconcile_seconds)

    def verified_fees(self) -> tuple[float, float]:
        payload = self.read_json(self.fee_state_path, {})
        if not payload.get("private_preflight_complete"):
            raise RuntimeError("Private fee preflight is incomplete.")
        if payload.get("profile") != PORTFOLIOS["FDUSD"].profile_name:
            raise RuntimeError("Private fee state belongs to a different credential profile.")
        maker = float(payload["maker_fee"])
        taker = float(payload["taker_fee"])
        fetched_at = int(payload.get("fetched_at", 0))
        if time.time() - fetched_at > 8 * 86400:
            raise RuntimeError("Private fee state is stale; refresh Binance account commission.")
        if maker < 0 or taker < 0:
            raise RuntimeError("Private fee values cannot be negative.")
        return maker, taker

    def target(self, state: dict, now_ts: int) -> tuple[str, int, int]:
        if not state:
            end_ts = now_ts // INTERVAL_SECONDS * INTERVAL_SECONDS
            return f"initial-30d-{end_ts}", 30, end_ts
        cutoff, _ = weekly_cutoff(now_ts)
        if str(state.get("period", "")).startswith("initial-") and cutoff <= int(state["train_end"]):
            return str(state["period"]), int(state["lookback_days"]), int(state["train_end"])
        return f"weekly-14d-{cutoff}", 14, cutoff

    def reconcile(self) -> None:
        self.ensure_staging()
        self.publish_macro_gate()
        if not self.parameter_updates_enabled:
            self.ensure_fixed_selection()
            return
        state = self.read_json(self.state_path, {})
        period, lookback_days, train_end = self.target(state, int(time.time()))
        if state.get("period") == period:
            return
        maker_fee, taker_fee = self.verified_fees()
        train_start = train_end - lookback_days * 86400
        candles = {
            pair: load_candles(pair, train_start, train_end, self.cache, allow_download=True)
            for pair in PORTFOLIOS["FDUSD"].pairs
        }
        selected, evaluations = select_candidate(candles, maker_fee)
        report_dir = self.selections / period
        report_dir.mkdir(parents=True, exist_ok=True)
        evaluations.to_csv(report_dir / "candidate_evaluations.csv", index=False)
        selection = {
            "schema_version": ACTIVE_SELECTION_SCHEMA_VERSION,
            "parameter_version": period,
            "generated_at": datetime.now(SHANGHAI).isoformat(),
            "valid_from": train_end,
            "training_window": {
                "lookback_days": lookback_days,
                "start_ts": train_start,
                "end_ts": train_end,
            },
            "trading_pairs": list(PORTFOLIOS["FDUSD"].pairs),
            "maker_fee": maker_fee,
            "taker_fee": taker_fee,
            "parameters": {
                "half_range": selected.half_range,
                "minimum_spread": selected.min_spread,
                "take_profit": selected.take_profit,
                "move_threshold": selected.move_threshold,
                "levels": selected.levels,
                "min_grid_move_seconds": selected.move_cooldown_seconds,
            },
        }
        self.atomic_json(report_dir / "selection.json", selection)
        self.atomic_json(self.canonical_selection, selection)
        self.write_disabled_config(candles, selected, maker_fee, period)
        updated_instances = self.publish_to_active_instances(selection)
        state = {
            "period": period,
            "lookback_days": lookback_days,
            "train_start": train_start,
            "train_end": train_end,
            "maker_fee": maker_fee,
            "taker_fee": taker_fee,
            "selection": str(report_dir / "selection.json"),
            "updated_instances": updated_instances,
            "trading_enabled": False,
        }
        self.atomic_json(self.state_path, state)
        LOG.warning("Published FDUSD parameter version %s to %d instance(s).", period, updated_instances)

    def ensure_fixed_selection(self) -> dict:
        selection = {
            "schema_version": ACTIVE_SELECTION_SCHEMA_VERSION,
            "parameter_version": "fixed-grid-6pct-ethbtc-forced-exit-v22",
            "generated_at": "2026-07-28T00:00:00+00:00",
            "valid_from": 0,
            "training_window": {
                "mode": "fixed-approved-structure",
                "source": "180d-parameter-search-2026-07-28",
            },
            "trading_pairs": list(PORTFOLIOS["FDUSD"].pairs),
            "maker_fee": None,
            "taker_fee": None,
            "parameters": {
                "half_range": 0.03,
                "minimum_spread": 0.006,
                "take_profit": 0.006,
                "move_threshold": 0.015,
                "levels": 10,
                "min_grid_move_seconds": 1800,
            },
            "technical_buy_gate": {
                "schema": XGBOOST_GATE_SCHEMA,
                "model_version": XGBOOST_MODEL_VERSION,
                "execution_policy_version": "v22-risk-off-forced-exit-v2",
                "combination": "long_only_per_pair",
                "short_spike_enabled": False,
                "mechanism1_runtime_fallback": False,
            },
        }
        current = self.read_json(self.canonical_selection, None)
        if current != selection:
            self.atomic_json(self.canonical_selection, selection)
        self.publish_to_active_instances(selection)
        state = {
            "mode": "fixed",
            "parameter_updates_enabled": False,
            "parameter_version": selection["parameter_version"],
            "trading_enabled": False,
        }
        if self.read_json(self.state_path, None) != state:
            self.atomic_json(self.state_path, state)
        return selection

    def write_disabled_config(self, candles: dict, selected, maker_fee: float, period: str) -> None:
        prices = {
            pair: Decimal(str(frame.close.iloc[-1]))
            for pair, frame in candles.items()
        }
        config = build_live_config(
            PORTFOLIOS["FDUSD"],
            prices,
            Decimal(str(maker_fee)),
            trading_enabled=False,
            bootstrap_from_quote=True,
            bootstrap_completed=False,
        )
        config.update({
            "grid_range": selected.half_range * 2,
            "grid_levels": selected.levels,
            "take_profit": float(effective_take_profit(
                Decimal(str(maker_fee)), Decimal(str(selected.take_profit))
            )),
            "move_threshold": selected.move_threshold,
            "min_grid_move_seconds": selected.move_cooldown_seconds,
            "active_parameter_version": period,
        })
        self.atomic_yaml(self.canonical_config, config)
        self.atomic_yaml(self.bots / "conf" / "scripts" / PORTFOLIOS["FDUSD"].config_name, config)

    def ensure_staging(self) -> None:
        scripts = self.bots / "scripts"
        configs = self.bots / "conf" / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        configs.mkdir(parents=True, exist_ok=True)
        shutil.copy2("/app/walk_forward_portfolio_grid_live.py", scripts / "walk_forward_portfolio_grid_live.py")
        shutil.copy2("/app/grid_live_common.py", scripts / "grid_live_common.py")
        shutil.copy2("/app/grid_macro_gate.py", scripts / "grid_macro_gate.py")
        shutil.copy2("/app/grid_xgboost_risk_gate.py", scripts / "grid_xgboost_risk_gate.py")

    def publish_macro_gate(self) -> int:
        macro_state = self.read_json(self.macro_source, None)
        gate = build_grid_macro_gate(
            macro_state,
            now=datetime.now(timezone.utc),
            max_source_age_seconds=self.macro_max_age_seconds,
            execution_enabled=self.macro_execution_enabled,
        )
        self.atomic_json(self.canonical_macro_gate, gate)
        instances = self.bots / "instances"
        updated = 0
        if instances.exists():
            for instance in instances.glob(f"{PORTFOLIOS['FDUSD'].bot_name}*"):
                if not instance.is_dir():
                    continue
                self.atomic_json(instance / "data" / "macro_gate.json", gate)
                updated += 1
        transition = (
            bool(gate["source_healthy"]),
            bool(gate["pause_new_orders"]),
            bool(gate["execution_enabled"]),
            tuple(gate["active_lease_ids"]),
            str(gate["reason"]),
        )
        if transition != self._last_macro_key:
            self._last_macro_key = transition
            LOG.warning(
                "Published FOMC gate healthy=%s enabled=%s paused=%s leases=%s reason=%s to %d instance(s).",
                gate["source_healthy"],
                gate["execution_enabled"],
                gate["pause_new_orders"],
                ",".join(gate["active_lease_ids"]) or "none",
                gate["reason"],
                updated,
            )
        return updated

    def publish_to_active_instances(self, selection: dict) -> int:
        instances = self.bots / "instances"
        if not instances.exists():
            return 0
        updated = 0
        for instance in instances.glob(f"{PORTFOLIOS['FDUSD'].bot_name}*"):
            if not instance.is_dir():
                continue
            runtime_state = instance / "data" / "live_grid_runtime_state.json"
            if not runtime_state.exists():
                continue
            self.atomic_json(instance / "data" / "active_selection.json", selection)
            updated += 1
        return updated


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    Scheduler().run_forever()
