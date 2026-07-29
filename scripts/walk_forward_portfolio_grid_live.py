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
from scripts.grid_technical_gate import load_runtime_technical_gate

RUNTIME_STATE_SCHEMA_VERSION = 4


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
    technical_buy_gate_file: str = "data/technical_buy_gate.json"
    technical_buy_gate_poll_seconds: int = 5
    technical_buy_gate_max_age_seconds: int = 150
    technical_buy_fail_closed: bool = True

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
        if self.risk_state_persist_seconds != RISK_STATE_PERSIST_SECONDS:
            raise ValueError("Live Grid risk state must be persisted every 5 seconds.")
        if set(self.reserved_base_by_pair) != set(self.trading_pairs):
            raise ValueError("Every pair requires a dedicated positive base reservation.")
        if any(value <= 0 for value in self.reserved_base_by_pair.values()):
            raise ValueError("Base reservations must be positive.")
        if self.trading_enabled and self.bootstrap_from_quote and not self.bootstrap_completed:
            raise ValueError("FDUSD inventory bootstrap must finish before live trading can be enabled.")
        if self.quote_asset == "FDUSD":
            if not self.macro_gate_enabled or not self.macro_fail_closed:
                raise ValueError("FDUSD live Grid requires a fail-closed FOMC macro gate.")
            if not 1 <= self.macro_gate_poll_seconds <= 30:
                raise ValueError("FOMC macro gate polling must be between 1 and 30 seconds.")
            if not 30 <= self.macro_gate_max_age_seconds <= 180:
                raise ValueError("FOMC macro gate freshness must be between 30 and 180 seconds.")
            if not self.technical_buy_gate_enabled or not self.technical_buy_fail_closed:
                raise ValueError("FDUSD live Grid requires a fail-closed ROC/SQZMOM BUY gate.")
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
        self.pending_flatten: set[str] = set()
        self.flatten_order_ids: Dict[str, str] = {}
        self.buy_order_ids: set[str] = set()
        self.sell_order_ids: set[str] = set()
        self.last_market_success = time.time()
        self.first_cycle_failure_at: Optional[float] = None
        self.disabled_logged = False
        self.peak_equity = config.capital_limit_quote
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
        self.technical_buy_enabled = not config.technical_buy_gate_enabled
        self.technical_gate_healthy = not config.technical_buy_gate_enabled
        self.technical_reason = (
            "technical_gate_not_checked" if config.technical_buy_gate_enabled else "disabled"
        )
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

            self._poll_technical_buy_gate()
            self._poll_macro_gate()
            if self.macro_paused:
                if self._owned_active_orders(active):
                    self.cancel_owned_orders()
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
            self.technical_buy_enabled = True
            self.technical_gate_healthy = True
            self.technical_reason = "disabled"
            return
        if self.current_timestamp < self.next_technical_poll:
            return
        self.next_technical_poll = (
            self.current_timestamp + self.config.technical_buy_gate_poll_seconds
        )
        gate = load_runtime_technical_gate(
            Path(self.config.technical_buy_gate_file),
            now=datetime.now(timezone.utc),
            max_age_seconds=self.config.technical_buy_gate_max_age_seconds,
        )
        healthy = bool(gate.get("runtime_gate_healthy"))
        enabled = bool(gate.get("buy_enabled"))
        if not healthy and not self.config.technical_buy_fail_closed:
            enabled = True
        reason = str(gate.get("reason", "unknown"))
        transition = (healthy, enabled, reason)
        previous_enabled = self.technical_buy_enabled
        self.technical_gate_healthy = healthy
        self.technical_buy_enabled = enabled
        self.technical_reason = reason
        self.technical_signal = dict(gate.get("signal", {}))
        if not enabled:
            unknown_side_state = self.cancel_owned_buy_orders()
            if unknown_side_state:
                self.next_refresh = 0.0
                self._record_runtime_event(
                    "technical_risk_off_unknown_sides_sell_rebuild_scheduled",
                    reason=reason,
                )
        elif not previous_enabled:
            self.next_refresh = 0.0
            self._record_runtime_event(
                "technical_buy_gate_recovered_immediate_refresh",
                reason=reason,
            )
        if transition != self.technical_transition_key:
            self.technical_transition_key = transition
            self.notify(
                f"ROC/SQZMOM BUY GATE {'ACTIVE' if enabled else 'RISK-OFF'}: "
                f"healthy={healthy} reason={reason}"
            )

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
            buy_budget = min(max(ledger.quote, Decimal("0")), self.config.side_budget_quote)
            if lower_levels and self.technical_buy_enabled:
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
                for level in upper_levels:
                    sell_price = max(level, price * (Decimal("1") + self.config.take_profit))
                    if amount * sell_price >= self.config.min_order_quote:
                        order_id = self.sell(self.config.exchange, pair, amount,
                                             OrderType.LIMIT_MAKER, sell_price)
                        ledger.open_order_ids.add(order_id)
                        self.sell_order_ids.add(order_id)

    def _control_risk(self, prices: Dict[str, Decimal]) -> None:
        total = self.config.reserve_quote
        for pair, ledger in self.ledgers.items():
            pair_equity = ledger.equity(prices[pair])
            total += pair_equity
            pair_pnl = pair_equity - self.config.pair_budget_quote
            ledger.peak_equity = max(ledger.peak_equity, pair_equity)
            pair_drawdown = (
                (ledger.peak_equity - pair_equity) / ledger.peak_equity
                if ledger.peak_equity > 0
                else Decimal("0")
            )
            pair_loss_tripped = pair_pnl <= -self.config.pair_stop_loss_quote
            pair_drawdown_tripped = pair_drawdown >= self.config.pair_drawdown_limit_pct
            if not ledger.halted and (pair_loss_tripped or pair_drawdown_tripped):
                ledger.halted = True
                self.pending_flatten.add(pair)
                reason = (
                    f"pnl={pair_pnl:.2f} <= -{self.config.pair_stop_loss_quote:.2f}"
                    if pair_loss_tripped
                    else f"drawdown={pair_drawdown:.2%}"
                )
                self.notify(
                    f"PAIR BREAKER {pair}: equity={pair_equity:.2f} "
                    f"peak={ledger.peak_equity:.2f} {reason}"
                )
        self.peak_equity = max(self.peak_equity, total)
        portfolio_pnl = total - self.config.capital_limit_quote
        portfolio_drawdown = (
            (self.peak_equity - total) / self.peak_equity
            if self.peak_equity > 0
            else Decimal("0")
        )
        portfolio_loss_tripped = portfolio_pnl <= -self.config.portfolio_stop_loss_quote
        portfolio_drawdown_tripped = (
            portfolio_drawdown >= self.config.portfolio_drawdown_limit_pct
        )
        if not self.portfolio_tripped and (
            portfolio_loss_tripped or portfolio_drawdown_tripped
        ):
            self.portfolio_tripped = True
            for ledger in self.ledgers.values():
                ledger.halted = True
            self.pending_flatten.update(self.config.trading_pairs)
            reason = (
                f"pnl={portfolio_pnl:.2f} <= -{self.config.portfolio_stop_loss_quote:.2f}"
                if portfolio_loss_tripped
                else f"drawdown={portfolio_drawdown:.2%}"
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
            delta = ledger.inventory_delta()
            if abs(delta) * prices[pair] < self.config.min_order_quote:
                completed.add(pair)
                continue
            if delta > 0:
                order_id = self.sell(self.config.exchange, pair, delta, OrderType.MARKET)
            else:
                order_id = self.buy(self.config.exchange, pair, abs(delta), OrderType.MARKET)
            ledger.open_order_ids.add(order_id)
            self.flatten_order_ids[pair] = order_id
        self.pending_flatten.difference_update(completed)

    def cancel_owned_orders(self, exclude: set[str] | None = None):
        exclude = exclude or set()
        for order in self._owned_active_orders(exclude=exclude):
            if order.client_order_id not in exclude:
                self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)

    def cancel_owned_buy_orders(self) -> bool:
        active = self._owned_active_orders()
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
            if schema_version not in {2, 3, RUNTIME_STATE_SCHEMA_VERSION}:
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
            self.pending_flatten = {
                pair for pair in state.get("pending_flatten", []) if pair in restored
            }
            self.flatten_order_ids = {
                pair: str(order_id)
                for pair, order_id in state.get("flatten_order_ids", {}).items()
                if pair in restored
            }
            self.buy_order_ids = {str(value) for value in state.get("buy_order_ids", [])}
            self.sell_order_ids = {str(value) for value in state.get("sell_order_ids", [])}
            self.peak_equity = Decimal(
                str(state.get("peak_equity", self.config.capital_limit_quote))
            )
            self.active_parameter_version = str(
                state.get("active_parameter_version", self.active_parameter_version)
            )
            self.runtime_events = [
                value for value in state.get("runtime_events", [])
                if isinstance(value, dict)
            ][-100:]
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
            "pending_flatten": sorted(self.pending_flatten),
            "flatten_order_ids": self.flatten_order_ids,
            "buy_order_ids": sorted(self.buy_order_ids),
            "sell_order_ids": sorted(self.sell_order_ids),
            "peak_equity": str(self.peak_equity),
            "active_parameter_version": self.active_parameter_version,
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
