"""Budget-isolated live variant of the paper moving portfolio grid.

This file deliberately does not replace ``walk_forward_portfolio_grid.py``. It
keeps the same moving-grid behavior while accounting only for explicitly
reserved quote and base inventory.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Field, field_validator, model_validator

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import MarketDict, OrderType, PriceType, TradeType
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    MarketOrderFailureEvent,
    OrderCancelledEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
)
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase

from scripts.grid_live_common import (
    BOT_LOSS_LIMIT,
    CAPITAL_LIMIT,
    MIN_ORDER_QUOTE,
    ORDER_REFRESH_SECONDS,
    PAIR_BUDGET,
    PAIR_DRAWDOWN_LIMIT_PCT,
    PAIR_LOSS_LIMIT,
    PORTFOLIO_DRAWDOWN_LIMIT_PCT,
    RISK_STATE_PERSIST_SECONDS,
    RESERVE_QUOTE,
    SIDE_BUDGET,
    STRATEGY_BUDGET,
    PairLedger,
    budget_for_quote,
    validate_active_selection,
)
from scripts.grid_macro_gate import load_runtime_macro_gate
from scripts.grid_xgboost_risk_gate import load_runtime_xgboost_gate
from scripts.risk_recovery import (
    ACTIVE, COOLDOWN, EXITING, LATCHED, REENTRY,
    advance_recovery, active_state, mark_exit_complete, mark_reentry_complete,
    normalize_state, trigger_state,
)

RUNTIME_STATE_SCHEMA_VERSION = 8


@dataclass
class GridState:
    lower: Decimal
    upper: Decimal
    levels: List[Decimal]
    moves: int = 0
    last_move_ts: float = 0


class LivePortfolioGridConfig(StrategyV2ConfigBase):
    script_file_name: str = os.path.basename(__file__)
    controllers_config: List[str] = []
    exchange: str = Field("binance")
    trading_pairs: List[str]
    quote_asset: str
    capital_limit_quote: Decimal = CAPITAL_LIMIT
    strategy_budget_quote: Decimal = STRATEGY_BUDGET
    reserve_quote: Decimal = RESERVE_QUOTE
    pair_budget_quote: Decimal = PAIR_BUDGET
    side_budget_quote: Decimal = SIDE_BUDGET
    reserved_base_by_pair: Dict[str, Decimal]
    grid_range: Decimal = Decimal("0.08")
    grid_levels: int = 8
    take_profit: Decimal = Decimal("0.008")
    move_threshold: Decimal = Decimal("0.02")
    min_grid_move_seconds: int = 1800
    order_refresh_time: int = ORDER_REFRESH_SECONDS
    pair_breakers_enabled: bool = True
    pair_loss_breaker_enabled: bool = True
    pair_drawdown_breaker_enabled: bool = True
    portfolio_breakers_enabled: bool = True
    portfolio_loss_breaker_enabled: bool = True
    portfolio_drawdown_breaker_enabled: bool = True
    cost_floor_enabled: bool = True
    inventory_exit_enabled: bool = True
    max_extra_inventory_quote: Decimal = Decimal("10")
    profit_protection_seconds: int = 24 * 60 * 60
    max_extra_inventory_hold_seconds: int = 48 * 60 * 60
    risk_state_persist_seconds: int = RISK_STATE_PERSIST_SECONDS
    portfolio_stop_loss_quote: Decimal = BOT_LOSS_LIMIT
    pair_stop_loss_quote: Decimal = PAIR_LOSS_LIMIT
    portfolio_drawdown_limit_pct: Decimal = PORTFOLIO_DRAWDOWN_LIMIT_PCT
    pair_drawdown_limit_pct: Decimal = PAIR_DRAWDOWN_LIMIT_PCT
    fail_closed_seconds: int = 60
    min_order_quote: Decimal = MIN_ORDER_QUOTE
    fee_rate: Decimal = Decimal("0.001")
    trading_enabled: bool = False
    bootstrap_from_quote: bool = False
    bootstrap_completed: bool = False
    active_selection_file: str = "data/active_selection.json"
    runtime_state_file: str = "data/live_grid_runtime_state.json"
    parameter_poll_seconds: int = 60
    active_parameter_version: str = "bootstrap-static-v1"
    macro_gate_enabled: bool = False
    macro_gate_file: str = "data/macro_gate.json"
    macro_gate_poll_seconds: int = 5
    macro_gate_max_age_seconds: int = 150
    macro_fail_closed: bool = True
    technical_buy_gate_enabled: bool = False
    technical_buy_gate_file: str = "data/xgboost_risk_gate.json"
    technical_buy_gate_poll_seconds: int = 5
    technical_buy_gate_max_age_seconds: int = 150
    technical_buy_fail_closed: bool = True
    technical_model_sha256: str = ""
    technical_feature_sha256: str = ""
    risk_auto_reentry_enabled: bool = False

    @field_validator("trading_pairs", mode="before")
    @classmethod
    def parse_pairs(cls, value):
        if isinstance(value, str):
            return [part.strip().upper() for part in value.split(",") if part.strip()]
        return value

    @model_validator(mode="after")
    def validate_safety(self):
        expected = [f"BTC-{self.quote_asset}", f"ETH-{self.quote_asset}"]
        budget = budget_for_quote(self.quote_asset)
        if self.exchange != "binance" or "perpetual" in self.exchange:
            raise ValueError("Live portfolio grid supports Binance spot only.")
        if self.trading_pairs != expected:
            raise ValueError(f"Expected exactly {expected}.")
        if (
            self.capital_limit_quote != budget.capital_limit
            or self.strategy_budget_quote != budget.strategy_budget
        ):
            raise ValueError("Live capital limits do not match the approved quote-asset budget.")
        if (
            self.reserve_quote != budget.reserve_quote
            or self.pair_budget_quote != budget.pair_budget
            or self.side_budget_quote != budget.side_budget
        ):
            raise ValueError("Live reserve and pair allocation do not match the approved budget.")
        if (
            self.portfolio_stop_loss_quote != budget.portfolio_loss_limit
            or self.pair_stop_loss_quote != budget.pair_loss_limit
        ):
            raise ValueError("Live stop losses do not match the approved risk budget.")
        if self.portfolio_drawdown_limit_pct != PORTFOLIO_DRAWDOWN_LIMIT_PCT:
            raise ValueError("Portfolio peak drawdown limit must be exactly 6%.")
        if self.pair_drawdown_limit_pct != PAIR_DRAWDOWN_LIMIT_PCT:
            raise ValueError("Pair peak drawdown limit must be exactly 3%.")
        if self.order_refresh_time != ORDER_REFRESH_SECONDS:
            raise ValueError("Live Grid order refresh time must be exactly 2 hours.")
        if self.inventory_exit_enabled:
            if self.max_extra_inventory_quote != Decimal("10"):
                raise ValueError("Extra inventory must be capped at exactly 10 quote units per pair.")
            if self.profit_protection_seconds != 24 * 60 * 60:
                raise ValueError("Inventory profit protection must last exactly 24 hours.")
            if self.max_extra_inventory_hold_seconds != 48 * 60 * 60:
                raise ValueError("Extra inventory maximum holding time must be exactly 48 hours.")
        if self.risk_state_persist_seconds != RISK_STATE_PERSIST_SECONDS:
            raise ValueError("Live Grid risk state must be persisted every 5 seconds.")
        if set(self.reserved_base_by_pair) != set(self.trading_pairs):
            raise ValueError("Every pair requires a dedicated positive base reservation.")
        if any(value <= 0 for value in self.reserved_base_by_pair.values()):
            raise ValueError("Base reservations must be positive.")
        if self.trading_enabled and self.bootstrap_from_quote and not self.bootstrap_completed:
            raise ValueError("FDUSD inventory bootstrap must finish before live trading can be enabled.")
        if self.quote_asset == "FDUSD":
            if self.macro_gate_enabled and not self.macro_fail_closed:
                raise ValueError("FDUSD live Grid requires a fail-closed FOMC macro gate.")
            if not 1 <= self.macro_gate_poll_seconds <= 30:
                raise ValueError("FOMC macro gate polling must be between 1 and 30 seconds.")
            if not 30 <= self.macro_gate_max_age_seconds <= 180:
                raise ValueError("FOMC macro gate freshness must be between 30 and 180 seconds.")
            if self.technical_buy_gate_enabled and not self.technical_buy_fail_closed:
                raise ValueError("FDUSD live Grid requires a fail-closed XGBoost BUY gate.")
            if Path(self.technical_buy_gate_file).name != "xgboost_risk_gate.json":
                raise ValueError("FDUSD live Grid does not permit a Mechanism 1 technical-gate file.")
            if self.trading_enabled and self.technical_buy_gate_enabled:
                hashes = (self.technical_model_sha256, self.technical_feature_sha256)
                if any(
                    len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower())
                    for value in hashes
                ):
                    raise ValueError("Enabled FDUSD Grid requires locked XGBoost model and feature hashes.")
            if not 1 <= self.technical_buy_gate_poll_seconds <= 30:
                raise ValueError("Technical BUY gate polling must be between 1 and 30 seconds.")
            if not 30 <= self.technical_buy_gate_max_age_seconds <= 180:
                raise ValueError("Technical BUY gate freshness must be between 30 and 180 seconds.")
        if self.grid_levels < 4 or self.grid_levels % 2:
            raise ValueError("grid_levels must be an even number of at least four.")
        return self

    def update_markets(self, markets: MarketDict) -> MarketDict:
        markets[self.exchange] = markets.get(self.exchange, set()) | set(self.trading_pairs)
        return markets


class LivePortfolioGrid(StrategyV2Base):
    def __init__(self, connectors: Dict[str, ConnectorBase], config: LivePortfolioGridConfig):
        super().__init__(connectors, config)
        self.config = config
        self.ledgers = {
            pair: PairLedger.create(pair, Decimal(str(config.reserved_base_by_pair[pair])))
            for pair in config.trading_pairs
        }
        self.grid_states: Dict[str, GridState] = {}
        self.next_refresh = 0.0
        self.next_risk_persist = 0.0
        self.awaiting_cancellation = False
        self.portfolio_tripped = False
        self.pair_recovery: Dict[str, Dict[str, Any]] = {
            pair: active_state() for pair in config.trading_pairs
        }
        self.portfolio_recovery: Dict[str, Any] = active_state()
        self.pending_flatten: set[str] = set()
        self.flatten_order_ids: Dict[str, str] = {}
        self.reentry_order_ids: Dict[str, str] = {}
        self.pending_inventory_exit: set[str] = set()
        self.inventory_exit_order_ids: Dict[str, str] = {}
        self.excess_inventory_started_at: Dict[str, Optional[float]] = {
            pair: None for pair in config.trading_pairs
        }
        self.buy_order_ids: set[str] = set()
        self.sell_order_ids: set[str] = set()
        self.last_market_success = time.time()
        self.first_cycle_failure_at: Optional[float] = None
        self.disabled_logged = False
        self.peak_equity = config.capital_limit_quote
        self.portfolio_episode_baseline = config.capital_limit_quote
        self.active_parameter_version = config.active_parameter_version
        self.pending_parameter_version: Optional[str] = None
        self.pending_parameters: Optional[Dict[str, Any]] = None
        self.next_parameter_poll = 0.0
        self.selection_mtime_ns = -1
        self.state_invalid_reason: Optional[str] = None
        self.macro_paused = bool(config.macro_gate_enabled and config.macro_fail_closed)
        self.macro_gate_healthy = not config.macro_gate_enabled
        self.macro_reason = "macro_gate_not_checked" if config.macro_gate_enabled else "disabled"
        self.macro_active_lease_ids: List[str] = []
        self.next_macro_poll = 0.0
        self.macro_transition_key: Optional[tuple] = None
        initial_enabled = not config.technical_buy_gate_enabled
        initial_healthy = not config.technical_buy_gate_enabled
        initial_reason = "xgboost_gate_not_checked" if config.technical_buy_gate_enabled else "disabled"
        self.technical_buy_enabled_by_pair = {
            pair: initial_enabled for pair in config.trading_pairs
        }
        self.technical_gate_healthy_by_pair = {
            pair: initial_healthy for pair in config.trading_pairs
        }
        self.technical_reason_by_pair = {
            pair: initial_reason for pair in config.trading_pairs
        }
        self.technical_signal_by_pair: Dict[str, Dict[str, Any]] = {
            pair: {} for pair in config.trading_pairs
        }
        # Aggregate compatibility fields are status-only. Order placement uses
        # the independent per-pair mappings above.
        self.technical_buy_enabled = all(self.technical_buy_enabled_by_pair.values())
        self.technical_gate_healthy = all(self.technical_gate_healthy_by_pair.values())
        self.technical_reason = initial_reason
        self.technical_signal: Dict[str, Any] = {}
        self.next_technical_poll = 0.0
        self.technical_transition_key: Optional[tuple] = None
        self.runtime_events: List[Dict[str, Any]] = []
        self._restore_state()

    @property
    def connector(self) -> ConnectorBase:
        return self.connectors[self.config.exchange]

    def on_tick(self):
        # The environment switch is a deployment-time interlock. Hummingbot API
        # instances do not inherit the manager container's environment, so the
        # immutable, validated per-instance config is the runtime authority.
        if not self.config.trading_enabled:
            self.cancel_owned_orders()
            if not self.disabled_logged:
                self.logger().warning("Live grid is disabled; owned orders are cancelled and no orders will be submitted.")
                self.disabled_logged = True
            return
        if self.config.bootstrap_from_quote and not self.config.bootstrap_completed:
            self.logger().error("Quote-only inventory bootstrap is incomplete; live orders are blocked.")
            self.cancel_owned_orders()
            return
        if self.state_invalid_reason:
            self.logger().error("Persistent live-grid state is invalid; fail-closed: %s", self.state_invalid_reason)
            self.cancel_owned_orders()
            return
        try:
            prices = {pair: self.reference_price(pair) for pair in self.config.trading_pairs}
            if any(price <= 0 for price in prices.values()):
                raise RuntimeError("A live reference price is unavailable.")
            self.last_market_success = self.current_timestamp
            self._control_risk(prices)
            active = self.get_active_orders(self.config.exchange)

            # Risk flattening always has priority over a macro pause.
            if self.pending_flatten:
                outstanding_grid, outstanding_flatten = self._partition_flatten_orders(active)
                self.cancel_owned_orders(exclude=set(self.flatten_order_ids.values()))
                if not outstanding_grid and not outstanding_flatten:
                    self._restore_inventory(prices)
                self._persist(prices)
                self.first_cycle_failure_at = None
                return

            # This timer-driven Taker action is independent of the XGBoost BUY
            # gate.  It can only sell inventory above the startup baseline.
            if self.config.inventory_exit_enabled and self._process_inventory_exit_policy(prices, active):
                self._persist(prices)
                self.first_cycle_failure_at = None
                return

            self._poll_technical_buy_gate()
            self._poll_macro_gate()
            if self.macro_paused:
                if self._owned_active_orders(active):
                    self.cancel_owned_orders()
                self._persist(prices)
                self.first_cycle_failure_at = None
                return

            if self._advance_risk_recovery(prices, active):
                self._persist(prices)
                self.first_cycle_failure_at = None
                return

            self._poll_parameter_update()
            if self.pending_parameters is not None:
                if self._owned_active_orders(active):
                    self.cancel_owned_orders()
                    self.next_refresh = self.current_timestamp + 5
                    self._persist(prices)
                    self.first_cycle_failure_at = None
                    return
                self._activate_pending_parameters()
            if self.portfolio_tripped:
                self._persist(prices)
                self.first_cycle_failure_at = None
                return
            if self.current_timestamp < self.next_refresh:
                if (
                    self.first_cycle_failure_at is not None
                    or self.current_timestamp >= self.next_risk_persist
                ):
                    self._persist(prices)
                    self.next_risk_persist = (
                        self.current_timestamp + self.config.risk_state_persist_seconds
                    )
                self.first_cycle_failure_at = None
                return
            if self._owned_active_orders(active):
                self.cancel_owned_orders()
                self.awaiting_cancellation = True
                self.next_refresh = self.current_timestamp + 5
            else:
                self.awaiting_cancellation = False
                self._place_grids(prices)
                self.next_refresh = self.current_timestamp + self.config.order_refresh_time
            self._persist(prices)
            self.next_risk_persist = self.current_timestamp + self.config.risk_state_persist_seconds
            self.first_cycle_failure_at = None
        except Exception as exc:
            self.logger().error("Live grid cycle failed: %s", exc)
            if self._cycle_failure_requires_trip(self.current_timestamp):
                self.portfolio_tripped = True
                self.pending_flatten.update(self.config.trading_pairs)
                self.cancel_owned_orders()

    def _poll_macro_gate(self) -> None:
        if not self.config.macro_gate_enabled:
            self.macro_paused = False
            self.macro_gate_healthy = True
            self.macro_reason = "disabled"
            return
        if self.current_timestamp < self.next_macro_poll:
            return
        self.next_macro_poll = self.current_timestamp + self.config.macro_gate_poll_seconds
        gate = load_runtime_macro_gate(
            Path(self.config.macro_gate_file),
            now=datetime.now(timezone.utc),
            max_age_seconds=self.config.macro_gate_max_age_seconds,
        )
        previous_paused = self.macro_paused
        healthy = bool(gate.get("runtime_gate_healthy"))
        paused = bool(gate.get("pause_new_orders"))
        if not healthy and not self.config.macro_fail_closed:
            paused = False
        lease_ids = sorted(str(value) for value in gate.get("active_lease_ids", []))
        reason = str(gate.get("reason", "unknown"))
        transition = (healthy, paused, tuple(lease_ids), reason)
        self.macro_gate_healthy = healthy
        self.macro_paused = paused
        self.macro_active_lease_ids = lease_ids
        self.macro_reason = reason
        if not healthy:
            self._latch_integrity_failure(f"FOMC contract unhealthy: {reason}")
        if previous_paused and not paused:
            self.next_refresh = 0.0
            self._record_runtime_event(
                "fomc_gate_resumed_immediate_refresh",
                reason=reason,
                active_lease_ids=lease_ids,
            )
        if transition != self.macro_transition_key:
            self.macro_transition_key = transition
            state = "PAUSED" if paused else "ACTIVE"
            self.notify(
                f"FOMC MACRO GATE {state}: healthy={healthy} "
                f"leases={','.join(lease_ids) or 'none'} reason={reason}"
            )

    def _poll_technical_buy_gate(self) -> None:
        if not self.config.technical_buy_gate_enabled:
            for pair in self.config.trading_pairs:
                self.technical_buy_enabled_by_pair[pair] = True
                self.technical_gate_healthy_by_pair[pair] = True
                self.technical_reason_by_pair[pair] = "disabled"
                self.technical_signal_by_pair[pair] = {}
            self.technical_buy_enabled = True
            self.technical_gate_healthy = True
            self.technical_reason = "disabled"
            return
        if self.current_timestamp < self.next_technical_poll:
            return
        self.next_technical_poll = (
            self.current_timestamp + self.config.technical_buy_gate_poll_seconds
        )
        gate = load_runtime_xgboost_gate(
            Path(self.config.technical_buy_gate_file),
            now=datetime.now(timezone.utc),
            max_age_seconds=self.config.technical_buy_gate_max_age_seconds,
            expected_model_sha256=self.config.technical_model_sha256 or None,
            expected_feature_sha256=self.config.technical_feature_sha256 or None,
        )
        healthy = bool(gate.get("runtime_gate_healthy"))
        reason = str(gate.get("reason", "unknown"))
        gate_pairs = gate.get("pairs", {})
        transitions = []
        for pair in self.config.trading_pairs:
            pair_signal = dict(gate_pairs.get(pair, {}))
            pair_healthy = bool(healthy and pair_signal)
            enabled = bool(pair_signal.get("buy_enabled")) if pair_healthy else False
            if not pair_healthy and not self.config.technical_buy_fail_closed:
                enabled = True
            pair_reason = str(pair_signal.get("reason", reason))
            previous_enabled = self.technical_buy_enabled_by_pair[pair]
            self.technical_gate_healthy_by_pair[pair] = pair_healthy
            self.technical_buy_enabled_by_pair[pair] = enabled
            self.technical_reason_by_pair[pair] = pair_reason
            self.technical_signal_by_pair[pair] = pair_signal
            transitions.append((pair, pair_healthy, enabled, pair_reason))
            forced_exit = bool(
                pair_healthy
                and gate.get("execution_policy_version") == "v22-risk-off-forced-exit-v2"
                and pair_signal.get("force_exit")
            )
            if (
                forced_exit
                and self.pair_recovery[pair].get("phase") == ACTIVE
            ):
                self.ledgers[pair].halted = True
                self.pending_flatten.add(pair)
                self.pair_recovery[pair] = trigger_state(
                    mechanism="v22_weekly_buy_gate", scope="technical",
                    now=self.current_timestamp,
                    trigger_value=pair_signal.get("probability"),
                    signal_price="", reason=pair_reason,
                )
                self.cancel_owned_orders()
                self._record_runtime_event(
                    "v22_forced_exit_triggered", pair=pair,
                    event_id=pair_signal.get("event_id"), reason=pair_reason,
                )
            elif not enabled:
                unknown_side_state = self.cancel_owned_buy_orders(pair)
                if unknown_side_state:
                    self.next_refresh = 0.0
                    self._record_runtime_event(
                        "xgboost_risk_off_unknown_sides_sell_rebuild_scheduled",
                        pair=pair, reason=pair_reason,
                    )
            elif not previous_enabled:
                self.next_refresh = 0.0
                self._record_runtime_event(
                    "xgboost_buy_gate_recovered_immediate_refresh",
                    pair=pair, reason=pair_reason,
                )
        self.technical_buy_enabled = all(self.technical_buy_enabled_by_pair.values())
        self.technical_gate_healthy = all(self.technical_gate_healthy_by_pair.values())
        self.technical_reason = reason
        self.technical_signal = {
            pair: dict(value) for pair, value in self.technical_signal_by_pair.items()
        }
        if not healthy:
            self._latch_integrity_failure(f"technical contract unhealthy: {reason}")
        transition = tuple(transitions)
        if transition != self.technical_transition_key:
            self.technical_transition_key = transition
            state = ",".join(
                f"{pair}:{'ACTIVE' if enabled else 'RISK-OFF'}"
                for pair, _, enabled, _ in transitions
            )
            self.notify(
                f"XGBOOST BUY GATE {state}: healthy={healthy} reason={reason}"
            )

    def _latch_integrity_failure(self, reason: str) -> None:
        if not hasattr(self, "portfolio_recovery"):
            self.portfolio_recovery = active_state()
        if self.portfolio_recovery.get("phase") == LATCHED:
            return
        self.portfolio_recovery = trigger_state(
            mechanism="infrastructure_integrity_breaker", scope="infrastructure",
            now=self.current_timestamp, trigger_value=reason, signal_price="",
            reason=reason, latch_after_exit=True,
        )
        self.portfolio_tripped = True
        for ledger in self.ledgers.values():
            ledger.halted = True
        self.pending_flatten.update(self.config.trading_pairs)
        self._record_runtime_event("integrity_failure_latched", reason=reason)
        self.notify(f"GRID INTEGRITY FAILURE LATCHED: {reason}")

    def _poll_parameter_update(self) -> None:
        if self.current_timestamp < self.next_parameter_poll:
            return
        self.next_parameter_poll = self.current_timestamp + self.config.parameter_poll_seconds
        target = Path(self.config.active_selection_file)
        if not target.exists():
            return
        stat = target.stat()
        if stat.st_mtime_ns == self.selection_mtime_ns:
            return
        self.selection_mtime_ns = stat.st_mtime_ns
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            parameters = validate_active_selection(payload, self.config.fee_rate)
            version = str(payload["parameter_version"])
            if version == self.active_parameter_version or version == self.pending_parameter_version:
                return
            self.pending_parameter_version = version
            self.pending_parameters = parameters
            self.logger().warning(
                "Validated parameter version %s; cancelling owned orders before activation.", version
            )
        except Exception as exc:
            self.logger().error("Rejected active selection update %s: %s", target, exc)

    def _activate_pending_parameters(self) -> None:
        if self.pending_parameters is None or self.pending_parameter_version is None:
            return
        params = self.pending_parameters
        self.config.grid_range = params["grid_range"]
        self.config.grid_levels = params["grid_levels"]
        self.config.take_profit = params["take_profit"]
        self.config.move_threshold = params["move_threshold"]
        self.config.min_grid_move_seconds = params["min_grid_move_seconds"]
        self.active_parameter_version = self.pending_parameter_version
        self.grid_states.clear()
        self.pending_parameters = None
        self.pending_parameter_version = None
        self.logger().warning("Activated live-grid parameter version %s.", self.active_parameter_version)

    def _place_grids(self, prices: Dict[str, Decimal]) -> None:
        for pair in self.config.trading_pairs:
            ledger = self.ledgers[pair]
            if ledger.halted:
                continue
            price = prices[pair]
            state = self.grid_states.get(pair)
            if state is None:
                state = self._new_grid(price)
                self.grid_states[pair] = state
            elif (price > state.upper * (Decimal("1") + self.config.move_threshold)
                  or price < state.lower * (Decimal("1") - self.config.move_threshold)):
                if self.current_timestamp - state.last_move_ts >= self.config.min_grid_move_seconds:
                    state = self._new_grid(price, state.moves + 1)
                    self.grid_states[pair] = state
            lower_levels = [level for level in state.levels if level < price]
            upper_levels = [level for level in state.levels if level > price]
            buy_limits = [max(ledger.quote, Decimal("0")), self.config.side_budget_quote]
            if self.config.inventory_exit_enabled:
                extra_quote = max(ledger.inventory_delta(), Decimal("0")) * price
                baseline_deficit_quote = max(-ledger.inventory_delta(), Decimal("0")) * price
                buy_limits.append(max(
                    baseline_deficit_quote + self.config.max_extra_inventory_quote - extra_quote,
                    Decimal("0"),
                ))
            buy_budget = min(buy_limits)
            if lower_levels and self.technical_buy_enabled_by_pair.get(pair, False):
                order_quote = buy_budget / Decimal(len(lower_levels))
                if order_quote >= self.config.min_order_quote:
                    for level in lower_levels:
                        order_id = self.buy(self.config.exchange, pair, order_quote / level,
                                            OrderType.LIMIT_MAKER, level)
                        ledger.open_order_ids.add(order_id)
                        self.buy_order_ids.add(order_id)
            sell_budget = min(max(ledger.base, Decimal("0")), ledger.initial_base)
            if upper_levels and sell_budget > 0:
                amount = sell_budget / Decimal(len(upper_levels))
                cost_floor = (
                    ledger.minimum_profitable_sell_price(
                        self._inventory_profit_floor_rate(pair)
                    )
                    if self.config.cost_floor_enabled
                    else Decimal("0")
                )
                for level in upper_levels:
                    sell_price = max(
                        level,
                        price * (Decimal("1") + self.config.take_profit),
                        cost_floor,
                    )
                    if amount * sell_price >= self.config.min_order_quote:
                        order_id = self.sell(self.config.exchange, pair, amount,
                                             OrderType.LIMIT_MAKER, sell_price)
                        ledger.open_order_ids.add(order_id)
                        self.sell_order_ids.add(order_id)

    def _inventory_profit_floor_rate(self, pair: str) -> Decimal:
        started = self.excess_inventory_started_at.get(pair)
        if started is None:
            return self.config.take_profit
        age = max(0.0, self.current_timestamp - float(started))
        return (
            self.config.take_profit
            if age < self.config.profit_protection_seconds else Decimal("0")
        )

    def _process_inventory_exit_policy(
        self, prices: Dict[str, Decimal], active_orders: list,
    ) -> bool:
        """Maintain extra-inventory timers and execute only the 48h excess exit."""
        active_ids = {order.client_order_id for order in active_orders}
        owned = self._owned_active_orders(active_orders)
        action_pending = False
        for pair, ledger in self.ledgers.items():
            extra = max(ledger.inventory_delta(), Decimal("0"))
            if extra <= 0:
                self.excess_inventory_started_at[pair] = None
                self.pending_inventory_exit.discard(pair)
                order_id = self.inventory_exit_order_ids.get(pair)
                if order_id is not None and order_id not in active_ids:
                    self.inventory_exit_order_ids.pop(pair, None)
                    ledger.open_order_ids.discard(order_id)
                continue
            if self.excess_inventory_started_at.get(pair) is None:
                self.excess_inventory_started_at[pair] = self.current_timestamp
            age = self.current_timestamp - float(self.excess_inventory_started_at[pair])
            if age < self.config.max_extra_inventory_hold_seconds:
                continue
            self.pending_inventory_exit.add(pair)
            order_id = self.inventory_exit_order_ids.get(pair)
            if order_id is not None and order_id in active_ids:
                action_pending = True
                continue
            if order_id is not None:
                self.inventory_exit_order_ids.pop(pair, None)
                ledger.open_order_ids.discard(order_id)
            pair_orders = [order for order in owned if order.trading_pair == pair]
            if pair_orders:
                for order in pair_orders:
                    self.cancel(self.config.exchange, pair, order.client_order_id)
                action_pending = True
                continue
            price = prices[pair]
            if extra * price < self.config.min_order_quote:
                self._record_runtime_event(
                    "inventory_48h_exit_below_exchange_minimum",
                    pair=pair, excess_quote=str(extra * price),
                )
                continue
            order_id = self.sell(self.config.exchange, pair, extra, OrderType.MARKET)
            ledger.open_order_ids.add(order_id)
            self.inventory_exit_order_ids[pair] = order_id
            self.sell_order_ids.add(order_id)
            self._record_runtime_event(
                "inventory_48h_excess_taker_exit",
                pair=pair, amount=str(extra), excess_quote=str(extra * price),
            )
            action_pending = True
        return action_pending

    def _control_risk(self, prices: Dict[str, Decimal]) -> None:
        # Compatibility for restored schema-7 states and lightweight test
        # fixtures constructed without __init__.
        if not hasattr(self, "pair_recovery"):
            self.pair_recovery = {pair: active_state() for pair in self.config.trading_pairs}
        if not hasattr(self, "portfolio_recovery"):
            self.portfolio_recovery = active_state()
        if not hasattr(self, "reentry_order_ids"):
            self.reentry_order_ids = {}
        if not hasattr(self, "portfolio_episode_baseline"):
            self.portfolio_episode_baseline = self.config.capital_limit_quote
        total = self.config.reserve_quote
        for pair, ledger in self.ledgers.items():
            pair_equity = ledger.equity(prices[pair])
            total += pair_equity
            pair_baseline = (
                ledger.episode_equity_baseline
                if ledger.episode_equity_baseline > 0 else self.config.pair_budget_quote
            )
            pair_pnl = pair_equity - pair_baseline
            ledger.peak_equity = max(ledger.peak_equity, pair_equity)
            pair_drawdown = (
                (ledger.peak_equity - pair_equity) / ledger.peak_equity
                if ledger.peak_equity > 0
                else Decimal("0")
            )
            pair_loss_tripped = (
                self.config.pair_breakers_enabled
                and getattr(self.config, "pair_loss_breaker_enabled", True)
                and pair_pnl <= -self.config.pair_stop_loss_quote
            )
            pair_drawdown_tripped = (
                self.config.pair_breakers_enabled
                and getattr(self.config, "pair_drawdown_breaker_enabled", True)
                and pair_drawdown >= self.config.pair_drawdown_limit_pct
            )
            if not ledger.halted and (pair_loss_tripped or pair_drawdown_tripped):
                reason = (
                    f"pnl={pair_pnl:.2f} <= -{self.config.pair_stop_loss_quote:.2f}"
                    if pair_loss_tripped
                    else f"drawdown={pair_drawdown:.2%}"
                )
                ledger.halted = True
                self.pending_flatten.add(pair)
                self.pair_recovery[pair] = trigger_state(
                    mechanism=("strategy_loss_breaker" if pair_loss_tripped
                               else "strategy_drawdown_breaker"),
                    scope="strategy", now=self.current_timestamp,
                    trigger_value=(pair_pnl if pair_loss_tripped else pair_drawdown),
                    signal_price=prices[pair], reason=reason,
                )
                self.notify(
                    f"PAIR BREAKER {pair}: equity={pair_equity:.2f} "
                    f"peak={ledger.peak_equity:.2f} {reason}"
                )
        self.peak_equity = max(self.peak_equity, total)
        portfolio_pnl = total - self.portfolio_episode_baseline
        portfolio_drawdown = (
            (self.peak_equity - total) / self.peak_equity
            if self.peak_equity > 0
            else Decimal("0")
        )
        portfolio_loss_tripped = (
            getattr(self.config, "portfolio_loss_breaker_enabled", True)
            and portfolio_pnl <= -self.config.portfolio_stop_loss_quote
        )
        portfolio_drawdown_tripped = (
            getattr(self.config, "portfolio_drawdown_breaker_enabled", True)
            and
            portfolio_drawdown >= self.config.portfolio_drawdown_limit_pct
        )
        if self.config.portfolio_breakers_enabled and not self.portfolio_tripped and (
            portfolio_loss_tripped or portfolio_drawdown_tripped
        ):
            reason = (
                f"pnl={portfolio_pnl:.2f} <= -{self.config.portfolio_stop_loss_quote:.2f}"
                if portfolio_loss_tripped
                else f"drawdown={portfolio_drawdown:.2%}"
            )
            self.portfolio_tripped = True
            for ledger in self.ledgers.values():
                ledger.halted = True
            self.pending_flatten.update(self.config.trading_pairs)
            self.portfolio_recovery = trigger_state(
                mechanism=("portfolio_loss_breaker" if portfolio_loss_tripped
                           else "portfolio_drawdown_breaker"),
                scope="portfolio", now=self.current_timestamp,
                trigger_value=(portfolio_pnl if portfolio_loss_tripped else portfolio_drawdown),
                signal_price={pair: str(prices[pair]) for pair in self.config.trading_pairs},
                reason=reason,
            )
            self.notify(
                f"PORTFOLIO BREAKER: equity={total:.2f} "
                f"peak={self.peak_equity:.2f} {reason}"
            )

    def _restore_inventory(self, prices: Dict[str, Decimal]) -> None:
        completed = set()
        for pair in self.pending_flatten:
            ledger = self.ledgers[pair]
            previous_flatten = self.flatten_order_ids.pop(pair, None)
            if previous_flatten:
                ledger.open_order_ids.discard(previous_flatten)
            # A stop is quote-only: liquidate all strategy-owned base rather
            # than returning to the startup inventory that can keep losing.
            delta = max(ledger.base, Decimal("0"))
            if delta * prices[pair] < self.config.min_order_quote:
                completed.add(pair)
                state = self.pair_recovery[pair]
                if self.portfolio_recovery.get("phase") == EXITING:
                    state = self.portfolio_recovery
                completed_state = mark_exit_complete(
                    state, now=self.current_timestamp,
                    remaining_base={pair: delta},
                    execution={
                        "target": "quote_only", "order_id": previous_flatten or "dust",
                        "attempts": int(state.get("exit_attempts", 0)),
                        "last_fill": state.get("last_exit_fill", {}),
                    },
                )
                if state is self.portfolio_recovery:
                    if all(
                        max(self.ledgers[item].base, Decimal("0")) * prices[item]
                        < self.config.min_order_quote
                        for item in self.config.trading_pairs
                    ):
                        self.portfolio_recovery = completed_state
                else:
                    self.pair_recovery[pair] = completed_state
                continue
            state = (
                self.portfolio_recovery
                if self.portfolio_recovery.get("phase") == EXITING
                else self.pair_recovery[pair]
            )
            state["first_exit_order_at"] = state.get("first_exit_order_at") or self.current_timestamp
            state["exit_attempts"] = int(state.get("exit_attempts", 0)) + 1
            order_id = self.sell(self.config.exchange, pair, delta, OrderType.MARKET)
            ledger.open_order_ids.add(order_id)
            self.flatten_order_ids[pair] = order_id
        self.pending_flatten.difference_update(completed)

    def _advance_risk_recovery(self, prices: Dict[str, Decimal], active_orders: list) -> bool:
        """Advance recoverable breakers; return True while normal grids must stay blocked."""
        active_owned = bool(self._owned_active_orders(active_orders))
        gates_healthy = bool(
            self.macro_gate_healthy
            and all(self.technical_gate_healthy_by_pair.values())
        )
        any_blocked = False
        for pair in self.config.trading_pairs:
            state = self.pair_recovery[pair]
            if state.get("phase") == ACTIVE:
                continue
            any_blocked = True
            gates_allow = bool(
                not self.macro_paused
                and self.technical_buy_enabled_by_pair.get(pair, False)
                and getattr(self.config, "risk_auto_reentry_enabled", False)
            )
            state = advance_recovery(
                state, now=self.current_timestamp,
                healthy=gates_healthy and not active_owned,
                gates_allow_reentry=gates_allow,
            )
            self.pair_recovery[pair] = state
            if state.get("reentry_allowed"):
                if self._reenter_pair(pair, prices[pair]):
                    self.pair_recovery[pair] = mark_reentry_complete(
                        state, now=self.current_timestamp,
                        baseline={"equity": self.ledgers[pair].equity(prices[pair])},
                    )
                    self.ledgers[pair].halted = False
                    self.ledgers[pair].episode_equity_baseline = self.ledgers[pair].equity(prices[pair])
        portfolio = self.portfolio_recovery
        if portfolio.get("phase") != ACTIVE:
            any_blocked = True
            gates_allow = bool(
                not self.macro_paused
                and all(self.technical_buy_enabled_by_pair.values())
                and getattr(self.config, "risk_auto_reentry_enabled", False)
            )
            portfolio = advance_recovery(
                portfolio, now=self.current_timestamp,
                healthy=gates_healthy and not active_owned,
                gates_allow_reentry=gates_allow,
            )
            self.portfolio_recovery = portfolio
            if portfolio.get("reentry_allowed"):
                completed = all(self._reenter_pair(pair, prices[pair]) for pair in self.config.trading_pairs)
                if completed:
                    equity = self.config.reserve_quote + sum(
                        ledger.equity(prices[pair]) for pair, ledger in self.ledgers.items()
                    )
                    self.portfolio_recovery = mark_reentry_complete(
                        portfolio, now=self.current_timestamp, baseline={"equity": equity},
                    )
                    self.portfolio_tripped = False
                    self.peak_equity = equity
                    self.portfolio_episode_baseline = equity
                    for pair, ledger in self.ledgers.items():
                        ledger.halted = False
                        ledger.peak_equity = ledger.equity(prices[pair])
                        ledger.episode_equity_baseline = ledger.equity(prices[pair])
                        self.pair_recovery[pair] = active_state()
        return any_blocked

    def _reenter_pair(self, pair: str, price: Decimal) -> bool:
        ledger = self.ledgers[pair]
        # Rebuild the configured quote-sized base inventory at the current
        # market price. The actual fill becomes the new owned risk baseline;
        # cumulative realised PnL is deliberately not reset.
        target = self.config.side_budget_quote / price
        missing = max(target - ledger.base, Decimal("0"))
        if missing * price < self.config.min_order_quote:
            ledger.peak_equity = ledger.equity(price)
            return True
        order_id = self.reentry_order_ids.get(pair)
        if order_id and order_id in self._owned_order_ids():
            return False
        order_id = self.buy(self.config.exchange, pair, missing, OrderType.MARKET)
        ledger.open_order_ids.add(order_id)
        self.reentry_order_ids[pair] = order_id
        self.buy_order_ids.add(order_id)
        self._record_runtime_event(
            "risk_reentry_market_buy", pair=pair, amount=str(missing), target_quote=str(self.config.side_budget_quote),
        )
        return False

    def cancel_owned_orders(self, exclude: set[str] | None = None):
        exclude = exclude or set()
        for order in self._owned_active_orders(exclude=exclude):
            if order.client_order_id not in exclude:
                self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)

    def cancel_owned_buy_orders(self, pair: str | None = None) -> bool:
        active = [
            order for order in self._owned_active_orders()
            if pair is None or order.trading_pair == pair
        ]
        # Old runtime-state schemas did not persist order sides. Cancelling all
        # owned orders once is safer than leaving an unknown BUY live; SELL is
        # rebuilt immediately at the next refresh.
        unknown_side_state = bool(active) and not self.buy_order_ids and not self.sell_order_ids
        for order in active:
            if unknown_side_state or order.client_order_id in self.buy_order_ids:
                self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)
        return unknown_side_state

    def _record_runtime_event(self, event: str, **details: Any) -> None:
        self.runtime_events.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **details,
        })
        self.runtime_events = self.runtime_events[-100:]

    def _owned_order_ids(self) -> set[str]:
        return set().union(*(ledger.open_order_ids for ledger in self.ledgers.values()))

    def _owned_active_orders(self, active=None, exclude: set[str] | None = None) -> list:
        active = self.get_active_orders(self.config.exchange) if active is None else active
        owned = self._owned_order_ids()
        excluded = exclude or set()
        return [
            order for order in active
            if order.client_order_id in owned and order.client_order_id not in excluded
        ]

    def _partition_flatten_orders(self, active) -> tuple[set[str], set[str]]:
        active_ids = {order.client_order_id for order in self._owned_active_orders(active)}
        flatten_ids = set(self.flatten_order_ids.values())
        return active_ids - flatten_ids, active_ids & flatten_ids

    def _cycle_failure_requires_trip(self, now: float) -> bool:
        if self.first_cycle_failure_at is None:
            self.first_cycle_failure_at = now
        return now - self.first_cycle_failure_at >= self.config.fail_closed_seconds

    def _forget_order(self, order_id: str) -> None:
        self.buy_order_ids.discard(order_id)
        self.sell_order_ids.discard(order_id)
        for pair, value in list(self.inventory_exit_order_ids.items()):
            if value == order_id:
                self.inventory_exit_order_ids.pop(pair, None)
        for ledger in self.ledgers.values():
            ledger.open_order_ids.discard(order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        ledger = self.ledgers.get(event.trading_pair)
        if ledger is None or event.order_id not in ledger.open_order_ids:
            return
        try:
            fee = event.trade_fee.fee_amount_in_token(
                event.trading_pair, event.price, event.amount, self.config.quote_asset
            )
        except Exception:
            fee = event.price * event.amount * self.config.fee_rate
        ledger.apply_fill(event.trade_type.name, event.price, event.amount, fee)
        flatten_pair = next(
            (pair for pair, order_id in self.flatten_order_ids.items() if order_id == event.order_id),
            None,
        )
        if flatten_pair is not None:
            state = (
                self.portfolio_recovery
                if self.portfolio_recovery.get("phase") == EXITING
                else self.pair_recovery[flatten_pair]
            )
            try:
                signal_price = Decimal(str(state.get("signal_price", event.price)))
            except Exception:
                signal_price = event.price
            slippage_bps = (
                (signal_price - event.price) / signal_price * Decimal("10000")
                if signal_price > 0 and event.trade_type == TradeType.SELL else Decimal("0")
            )
            state["last_exit_fill"] = {
                "filled_at": self.current_timestamp, "price": str(event.price),
                "amount": str(event.amount), "fee_quote": str(fee),
                "slippage_bps": str(slippage_bps),
                "signal_to_fill_seconds": self.current_timestamp - float(state.get("triggered_at") or self.current_timestamp),
            }
        if event.trade_type == TradeType.BUY:
            base_asset = event.trading_pair.split("-")[0]
            ledger.base -= sum(
                (flat_fee.amount for flat_fee in event.trade_fee.flat_fees if flat_fee.token == base_asset),
                Decimal("0"),
            )
        self.notify(f"{event.trade_type.name} {event.amount:.8f} {event.trading_pair} at {event.price:.8f}")

    def did_cancel_order(self, event: OrderCancelledEvent):
        self._forget_order(event.order_id)

    def did_expire_order(self, event: OrderExpiredEvent):
        self._forget_order(event.order_id)

    def did_fail_order(self, event: MarketOrderFailureEvent):
        self._forget_order(event.order_id)

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        self._forget_order(event.order_id)

    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        self._forget_order(event.order_id)

    def reference_price(self, pair: str) -> Decimal:
        price = self.connector.get_price_by_type(pair, PriceType.LastTrade)
        if price is None or price.is_nan() or price <= 0:
            price = self.connector.get_price_by_type(pair, PriceType.MidPrice)
        return price if price is not None and not price.is_nan() else Decimal("0")

    def _new_grid(self, center: Decimal, moves: int = 0) -> GridState:
        lower = center * (Decimal("1") - self.config.grid_range / Decimal("2"))
        upper = center * (Decimal("1") + self.config.grid_range / Decimal("2"))
        step = (upper - lower) / Decimal(self.config.grid_levels - 1)
        levels = [lower + step * Decimal(index) for index in range(self.config.grid_levels)]
        return GridState(lower, upper, levels, moves, self.current_timestamp)

    def _restore_state(self) -> None:
        target = Path(self.config.runtime_state_file)
        if not target.exists():
            return
        try:
            state = json.loads(target.read_text(encoding="utf-8"))
            schema_version = int(state.get("schema_version", -1))
            if schema_version not in {2, 3, 4, 5, 6, 7, RUNTIME_STATE_SCHEMA_VERSION}:
                raise ValueError("runtime state schema version mismatch")
            if tuple(state.get("trading_pairs", ())) != tuple(self.config.trading_pairs):
                raise ValueError("runtime state trading pairs mismatch")
            ledgers_payload = state.get("ledgers", {})
            if set(ledgers_payload) != set(self.config.trading_pairs):
                raise ValueError("runtime state ledgers do not match configured pairs")
            restored = {
                pair: PairLedger.from_mapping(ledgers_payload[pair])
                for pair in self.config.trading_pairs
            }
            for pair, ledger in restored.items():
                configured_base = Decimal(str(self.config.reserved_base_by_pair[pair]))
                if ledger.initial_base != configured_base:
                    raise ValueError(f"runtime inventory baseline mismatch for {pair}")
                if ledger.initial_quote != self.config.side_budget_quote:
                    raise ValueError(f"runtime quote baseline mismatch for {pair}")
            grid_states: Dict[str, GridState] = {}
            for pair, raw in state.get("grid_states", {}).items():
                if pair not in restored:
                    raise ValueError(f"unexpected grid state for {pair}")
                grid_states[pair] = GridState(
                    lower=Decimal(str(raw["lower"])),
                    upper=Decimal(str(raw["upper"])),
                    levels=[Decimal(str(value)) for value in raw["levels"]],
                    moves=int(raw.get("moves", 0)),
                    last_move_ts=float(raw.get("last_move_ts", 0)),
                )
            self.ledgers = restored
            self.grid_states = grid_states
            self.portfolio_tripped = bool(state.get("portfolio_tripped", False))
            self.pair_recovery = {
                pair: normalize_state(state.get("pair_recovery", {}).get(pair))
                for pair in restored
            }
            self.portfolio_recovery = normalize_state(state.get("portfolio_recovery"))
            self.pending_flatten = {
                pair for pair in state.get("pending_flatten", []) if pair in restored
            }
            self.flatten_order_ids = {
                pair: str(order_id)
                for pair, order_id in state.get("flatten_order_ids", {}).items()
                if pair in restored
            }
            self.reentry_order_ids = {
                pair: str(order_id)
                for pair, order_id in state.get("reentry_order_ids", {}).items()
                if pair in restored
            }
            self.pending_inventory_exit = {
                pair for pair in state.get("pending_inventory_exit", []) if pair in restored
            }
            self.inventory_exit_order_ids = {
                pair: str(order_id)
                for pair, order_id in state.get("inventory_exit_order_ids", {}).items()
                if pair in restored
            }
            saved_timers = state.get("excess_inventory_started_at", {})
            self.excess_inventory_started_at = {
                pair: (
                    float(saved_timers[pair]) if saved_timers.get(pair) is not None else None
                ) for pair in restored
            }
            self.buy_order_ids = {str(value) for value in state.get("buy_order_ids", [])}
            self.sell_order_ids = {str(value) for value in state.get("sell_order_ids", [])}
            self.peak_equity = Decimal(
                str(state.get("peak_equity", self.config.capital_limit_quote))
            )
            self.portfolio_episode_baseline = Decimal(str(
                state.get("portfolio_episode_baseline", self.config.capital_limit_quote)
            ))
            self.active_parameter_version = str(
                state.get("active_parameter_version", self.active_parameter_version)
            )
            self.runtime_events = [
                value for value in state.get("runtime_events", [])
                if isinstance(value, dict)
            ][-100:]
            saved_gate = state.get("technical_buy_gate", {})
            saved_pairs = saved_gate.get("pairs", {}) if isinstance(saved_gate, dict) else {}
            if schema_version == RUNTIME_STATE_SCHEMA_VERSION and set(saved_pairs) == set(restored):
                for pair in restored:
                    raw = saved_pairs[pair]
                    self.technical_buy_enabled_by_pair[pair] = bool(raw.get("buy_enabled", False))
                    self.technical_gate_healthy_by_pair[pair] = bool(raw.get("healthy", False))
                    self.technical_reason_by_pair[pair] = str(raw.get("reason", "restored_fail_closed"))
                    self.technical_signal_by_pair[pair] = dict(raw.get("signal", {}))
                self.technical_buy_enabled = all(self.technical_buy_enabled_by_pair.values())
                self.technical_gate_healthy = all(self.technical_gate_healthy_by_pair.values())
            saved_parameters = state.get("active_parameters", {})
            if saved_parameters:
                self.config.grid_range = Decimal(str(saved_parameters["grid_range"]))
                self.config.grid_levels = int(saved_parameters["grid_levels"])
                self.config.take_profit = Decimal(str(saved_parameters["take_profit"]))
                self.config.move_threshold = Decimal(str(saved_parameters["move_threshold"]))
                self.config.min_grid_move_seconds = int(saved_parameters["min_grid_move_seconds"])
        except Exception as exc:
            self.state_invalid_reason = str(exc)

    def _persist(self, prices: Dict[str, Decimal]) -> None:
        state = {
            "schema_version": RUNTIME_STATE_SCHEMA_VERSION,
            "trading_pairs": list(self.config.trading_pairs),
            "updated_at": self.current_timestamp,
            "portfolio_tripped": self.portfolio_tripped,
            "pair_recovery": self.pair_recovery,
            "portfolio_recovery": self.portfolio_recovery,
            "pending_flatten": sorted(self.pending_flatten),
            "flatten_order_ids": self.flatten_order_ids,
            "reentry_order_ids": self.reentry_order_ids,
            "pending_inventory_exit": sorted(self.pending_inventory_exit),
            "inventory_exit_order_ids": self.inventory_exit_order_ids,
            "excess_inventory_started_at": self.excess_inventory_started_at,
            "buy_order_ids": sorted(self.buy_order_ids),
            "sell_order_ids": sorted(self.sell_order_ids),
            "peak_equity": str(self.peak_equity),
            "portfolio_episode_baseline": str(self.portfolio_episode_baseline),
            "active_parameter_version": self.active_parameter_version,
            "cost_floor_enabled": self.config.cost_floor_enabled,
            "pair_breakers_enabled": self.config.pair_breakers_enabled,
            "portfolio_breakers_enabled": self.config.portfolio_breakers_enabled,
            "inventory_exit_enabled": self.config.inventory_exit_enabled,
            "inventory_exit_policy": {
                "max_extra_inventory_quote": str(self.config.max_extra_inventory_quote),
                "profit_protection_seconds": self.config.profit_protection_seconds,
                "max_hold_seconds": self.config.max_extra_inventory_hold_seconds,
            },
            "runtime_events": self.runtime_events,
            "macro_gate": {
                "enabled": self.config.macro_gate_enabled,
                "healthy": self.macro_gate_healthy,
                "paused": self.macro_paused,
                "reason": self.macro_reason,
                "active_lease_ids": self.macro_active_lease_ids,
            },
            "technical_buy_gate": {
                "enabled": self.config.technical_buy_gate_enabled,
                "healthy": self.technical_gate_healthy,
                "buy_enabled": self.technical_buy_enabled,
                "reason": self.technical_reason,
                "signal": self.technical_signal,
                "pairs": {
                    pair: {
                        "healthy": self.technical_gate_healthy_by_pair[pair],
                        "buy_enabled": self.technical_buy_enabled_by_pair[pair],
                        "reason": self.technical_reason_by_pair[pair],
                        "signal": self.technical_signal_by_pair[pair],
                    } for pair in self.config.trading_pairs
                },
            },
            "active_parameters": {
                "grid_range": str(self.config.grid_range),
                "grid_levels": self.config.grid_levels,
                "take_profit": str(self.config.take_profit),
                "move_threshold": str(self.config.move_threshold),
                "min_grid_move_seconds": self.config.min_grid_move_seconds,
            },
            "ledgers": {
                pair: {
                    **{key: str(value) if isinstance(value, Decimal) else value
                       for key, value in asdict(ledger).items() if key != "open_order_ids"},
                    "open_order_ids": sorted(ledger.open_order_ids),
                }
                for pair, ledger in self.ledgers.items()
            },
            "grid_states": {
                pair: {
                    "lower": str(grid.lower),
                    "upper": str(grid.upper),
                    "levels": [str(value) for value in grid.levels],
                    "moves": grid.moves,
                    "last_move_ts": grid.last_move_ts,
                }
                for pair, grid in self.grid_states.items()
            },
            "prices": {pair: str(price) for pair, price in prices.items()},
        }
        target = Path(self.config.runtime_state_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as output:
            json.dump(state, output, indent=2, ensure_ascii=True)
            temporary = output.name
        Path(temporary).replace(target)

    def notify(self, message: str):
        self.log_with_clock(logging.WARNING, message)
        self.notify_hb_app_with_timestamp(message)
