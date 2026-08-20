"""Shared constants, configuration, and accounting for the live grids."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence


CONNECTOR = "binance"


def _risk_enabled(name: str) -> bool:
    return os.getenv(name, "true").lower() == "true"


@dataclass(frozen=True)
class BudgetLimits:
    capital_limit: Decimal
    strategy_budget: Decimal
    reserve_quote: Decimal
    pair_budget: Decimal
    side_budget: Decimal
    pair_loss_limit: Decimal
    portfolio_loss_limit: Decimal
    recommended_balance: Decimal


USDT_BUDGET = BudgetLimits(
    capital_limit=Decimal("500"),
    strategy_budget=Decimal("475"),
    reserve_quote=Decimal("25"),
    pair_budget=Decimal("237.5"),
    side_budget=Decimal("118.75"),
    pair_loss_limit=Decimal("20"),
    portfolio_loss_limit=Decimal("40"),
    recommended_balance=Decimal("525"),
)
FDUSD_BUDGET = BudgetLimits(
    capital_limit=Decimal("420"),
    strategy_budget=Decimal("400"),
    reserve_quote=Decimal("20"),
    pair_budget=Decimal("200"),
    side_budget=Decimal("100"),
    pair_loss_limit=Decimal("6"),
    portfolio_loss_limit=Decimal("24"),
    recommended_balance=Decimal("440"),
)

# Backwards-compatible aliases for the pre-existing USDT validation path.
CAPITAL_LIMIT = USDT_BUDGET.capital_limit
STRATEGY_BUDGET = USDT_BUDGET.strategy_budget
RESERVE_QUOTE = USDT_BUDGET.reserve_quote
PAIR_BUDGET = USDT_BUDGET.pair_budget
SIDE_BUDGET = USDT_BUDGET.side_budget
PAIR_LOSS_LIMIT = USDT_BUDGET.pair_loss_limit
BOT_LOSS_LIMIT = USDT_BUDGET.portfolio_loss_limit
COMBINED_LOSS_LIMIT = Decimal("80")
MIN_ORDER_QUOTE = Decimal("5.25")
ORDER_REFRESH_SECONDS = 2 * 60 * 60
RISK_STATE_PERSIST_SECONDS = 5
STARTUP_ORDER_RECONCILE_SECONDS = 30
PAIR_DRAWDOWN_LIMIT_PCT = Decimal("0.03")
PORTFOLIO_DRAWDOWN_LIMIT_PCT = Decimal("0.06")
CONFIRMATION = "LIVE-GRID-FDUSD-400"
FDUSD_RECOMMENDED_BALANCE = FDUSD_BUDGET.recommended_balance
ACTIVE_SELECTION_SCHEMA_VERSION = 2
SUPPORTED_ACTIVE_SELECTION_SCHEMA_VERSIONS = frozenset({1, 2})
ALLOWED_HALF_RANGES = (Decimal("0.03"), Decimal("0.04"), Decimal("0.05"))
ALLOWED_MIN_SPREADS = (Decimal("0.006"), Decimal("0.008"), Decimal("0.010"))
ALLOWED_TAKE_PROFITS = (Decimal("0.006"), Decimal("0.008"), Decimal("0.010"))
ALLOWED_MOVE_THRESHOLDS = (Decimal("0.015"), Decimal("0.020"), Decimal("0.030"))
GRID_MOVE_COOLDOWN_SECONDS = 1800

# Immutable production profiles.  Schema-v2 contracts must match these values
# exactly; a profile name is not an escape hatch for arbitrary live settings.
BINANCE_AI_GRID_PROFILES: Dict[str, Dict[str, Decimal | int]] = {
    "medium_sideways": {
        "grid_range": Decimal("0.12698379475402316"),
        "grid_levels": 18,
        "take_profit": Decimal("0.004"),
        "minimum_order_quote": Decimal("10"),
        "move_threshold": Decimal("0.015"),
        "min_grid_move_seconds": 1800,
        "order_refresh_seconds": ORDER_REFRESH_SECONDS,
    },
    "long_volatility": {
        "grid_range": Decimal("0.5246511596640915"),
        "grid_levels": 18,
        "take_profit": Decimal("0.014179761072002472"),
        "minimum_order_quote": Decimal("10"),
        "move_threshold": Decimal("0.015"),
        "min_grid_move_seconds": 1800,
        "order_refresh_seconds": ORDER_REFRESH_SECONDS,
    },
}
APPROVED_PAIR_PROFILES = {
    "BTC-FDUSD": "medium_sideways",
    "ETH-FDUSD": "long_volatility",
}


@dataclass(frozen=True)
class GridPortfolio:
    quote_asset: str
    bot_name: str
    profile_name: str
    config_name: str
    pairs: tuple[str, str]


PORTFOLIOS: Dict[str, GridPortfolio] = {
    "USDT": GridPortfolio(
        quote_asset="USDT",
        bot_name="grid-live-usdt-500",
        profile_name="binance_live_grid_usdt_500",
        config_name="walk_forward_portfolio_grid_live_usdt_500.yml",
        pairs=("BTC-USDT", "ETH-USDT"),
    ),
    "FDUSD": GridPortfolio(
        quote_asset="FDUSD",
        bot_name="grid-live-fdusd-400",
        profile_name="binance_live_grid_fdusd_400",
        config_name="walk_forward_portfolio_grid_live_fdusd_400.yml",
        pairs=("BTC-FDUSD", "ETH-FDUSD"),
    ),
}


def budget_for_quote(quote_asset: str) -> BudgetLimits:
    quote = quote_asset.upper()
    if quote == "FDUSD":
        return FDUSD_BUDGET
    if quote == "USDT":
        return USDT_BUDGET
    raise ValueError(f"Unsupported live-grid quote asset: {quote_asset}")


def budget_for_pair(trading_pair: str) -> BudgetLimits:
    return budget_for_quote(trading_pair.rsplit("-", 1)[-1])


@dataclass
class PairLedger:
    trading_pair: str
    initial_quote: Decimal
    initial_base: Decimal
    quote: Decimal
    base: Decimal
    base_cost_quote: Decimal
    fees_quote: Decimal = Decimal("0")
    buys: int = 0
    sells: int = 0
    halted: bool = False
    open_order_ids: set[str] = field(default_factory=set)
    peak_equity: Decimal = Decimal("0")
    episode_equity_baseline: Decimal = Decimal("0")

    @classmethod
    def create(cls, trading_pair: str, initial_base: Decimal) -> "PairLedger":
        side_budget = budget_for_pair(trading_pair).side_budget
        pair_budget = budget_for_pair(trading_pair).pair_budget
        return cls(
            trading_pair,
            side_budget,
            initial_base,
            side_budget,
            initial_base,
            side_budget,
            peak_equity=pair_budget,
            episode_equity_baseline=pair_budget,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PairLedger":
        initial_quote = Decimal(str(payload["initial_quote"]))
        initial_base = Decimal(str(payload["initial_base"]))
        base = Decimal(str(payload["base"]))
        # Schema <= 4 did not persist inventory cost. Preserve compatibility by
        # valuing the restored inventory at the bootstrap unit cost. New fills
        # then maintain the exact moving-average cost from this migration point.
        migrated_cost = (
            initial_quote / initial_base * max(base, Decimal("0"))
            if initial_base > 0
            else Decimal("0")
        )
        return cls(
            trading_pair=str(payload["trading_pair"]),
            initial_quote=initial_quote,
            initial_base=initial_base,
            quote=Decimal(str(payload["quote"])),
            base=base,
            base_cost_quote=Decimal(str(payload.get("base_cost_quote", migrated_cost))),
            fees_quote=Decimal(str(payload.get("fees_quote", "0"))),
            buys=int(payload.get("buys", 0)),
            sells=int(payload.get("sells", 0)),
            halted=bool(payload.get("halted", False)),
            open_order_ids={str(value) for value in payload.get("open_order_ids", [])},
            peak_equity=Decimal(
                str(payload.get("peak_equity", budget_for_pair(str(payload["trading_pair"])).pair_budget))
            ),
            episode_equity_baseline=Decimal(str(payload.get(
                "episode_equity_baseline",
                budget_for_pair(str(payload["trading_pair"])).pair_budget,
            ))),
        )

    def equity(self, price: Decimal) -> Decimal:
        return self.quote + self.base * price

    def pnl(self, price: Decimal) -> Decimal:
        return self.equity(price) - budget_for_pair(self.trading_pair).pair_budget

    def apply_fill(self, side: str, price: Decimal, amount: Decimal, fee_quote: Decimal) -> None:
        notional = price * amount
        if side.upper() == "BUY":
            self.quote -= notional + fee_quote
            self.base += amount
            self.base_cost_quote += notional + fee_quote
            self.buys += 1
        elif side.upper() == "SELL":
            if self.base > 0:
                remaining = max(self.base - amount, Decimal("0"))
                self.base_cost_quote *= remaining / self.base
            self.quote += notional - fee_quote
            self.base -= amount
            self.sells += 1
        else:
            raise ValueError(f"Unsupported trade side: {side}")
        self.fees_quote += fee_quote

    def inventory_delta(self) -> Decimal:
        return self.base - self.initial_base

    def average_base_cost(self) -> Decimal:
        if self.base <= 0:
            return Decimal("0")
        return max(self.base_cost_quote, Decimal("0")) / self.base

    def minimum_profitable_sell_price(self, minimum_profit_rate: Decimal) -> Decimal:
        if minimum_profit_rate < 0:
            raise ValueError("minimum_profit_rate must be non-negative")
        return self.average_base_cost() * (Decimal("1") + minimum_profit_rate)


def effective_take_profit(maker_rate: Decimal, configured: Decimal = Decimal("0.008")) -> Decimal:
    return max(configured, maker_rate * Decimal("2") + Decimal("0.004"))


def clip_quantized_sell_levels(
    levels: Sequence[Decimal],
    sell_budget: Decimal,
    minimum_order_quote: Decimal,
    price_for_level: Callable[[Decimal], Decimal],
    quantize_amount: Callable[[Decimal], Decimal],
) -> list[tuple[Decimal, Decimal]]:
    """Remove far SELL levels until every quantized order is executable."""
    candidates = sorted(Decimal(str(level)) for level in levels)
    while candidates:
        amount = max(
            Decimal(str(quantize_amount(sell_budget / Decimal(len(candidates))))),
            Decimal("0"),
        )
        built = [(Decimal(str(price_for_level(level))), amount) for level in candidates]
        if (
            amount > 0
            and amount * Decimal(len(built)) <= sell_budget
            and all(price * quantity >= minimum_order_quote for price, quantity in built)
        ):
            return built
        candidates.pop()
    return []


def clip_quantized_buy_levels(
    levels: Sequence[Decimal],
    buy_budget_quote: Decimal,
    minimum_order_quote: Decimal,
    price_for_level: Callable[[Decimal], Decimal],
    quantize_amount: Callable[[Decimal], Decimal],
    *,
    amount_step: Decimal = Decimal("0"),
    minimum_amount: Decimal = Decimal("0"),
) -> list[tuple[Decimal, Decimal]]:
    """Build the nearest executable BUY levels without exceeding the budget.

    Binance's amount quantizer rounds quantities down.  When the budget is an
    exact multiple of the minimum notional this can turn every nominally valid
    order into an order just below the exchange minimum.  BUY quantities are
    therefore rounded *up* to the amount step before the connector performs its
    final quantization.  If the rounded orders do not fit, the farthest level is
    removed and the remaining budget is redistributed.

    SELL orders deliberately keep using :func:`clip_quantized_sell_levels` and
    downward quantization so this helper can never create an oversell.
    """
    budget = max(Decimal(str(buy_budget_quote)), Decimal("0"))
    minimum_quote = max(Decimal(str(minimum_order_quote)), Decimal("0"))
    step = max(Decimal(str(amount_step)), Decimal("0"))
    min_amount = max(Decimal(str(minimum_amount)), Decimal("0"))
    if budget <= 0 or minimum_quote <= 0 or budget < minimum_quote:
        return []

    # Lower Grid levels arrive below the market.  Descending order keeps the
    # closest price first; reducing the candidate count removes the farthest.
    candidates = sorted((Decimal(str(level)) for level in levels), reverse=True)
    candidates = candidates[:min(len(candidates), int(budget // minimum_quote))]
    initial_count = len(candidates)
    if initial_count == 0:
        return []
    # Keep the original per-layer allocation while clipping. Re-dividing the
    # entire budget after every removal would make an upward-rounded final
    # layer exceed the budget forever (including when only one layer remains).
    per_order_budget = budget / Decimal(initial_count)
    while candidates:
        built: list[tuple[Decimal, Decimal]] = []
        valid = True
        for level in candidates:
            price = Decimal(str(price_for_level(level)))
            if price <= 0:
                valid = False
                break
            raw_amount = max(per_order_budget / price, minimum_quote / price, min_amount)
            if step > 0:
                raw_amount = (
                    (raw_amount / step).to_integral_value(rounding=ROUND_CEILING) * step
                )
            amount = max(Decimal(str(quantize_amount(raw_amount))), Decimal("0"))
            # A connector may apply a second downward normalization.  Advance
            # one exchange step at a time until the actual quantized order is
            # executable.  The budget check below still prevents overspending.
            attempts = 0
            while (
                step > 0
                and amount > 0
                and (amount < min_amount or price * amount < minimum_quote)
                and attempts < 4
            ):
                raw_amount += step
                amount = max(Decimal(str(quantize_amount(raw_amount))), Decimal("0"))
                attempts += 1
            if amount <= 0 or amount < min_amount or price * amount < minimum_quote:
                valid = False
                break
            built.append((price, amount))
        total_quote = sum((price * amount for price, amount in built), Decimal("0"))
        if valid and len(built) == len(candidates) and total_quote <= budget:
            return built
        candidates.pop()
    return []


def required_balances(prices: Mapping[str, Decimal], quote_only_fdusd: bool = False) -> Dict[str, Decimal]:
    if quote_only_fdusd:
        return {"FDUSD": FDUSD_BUDGET.capital_limit}
    requirements: Dict[str, Decimal] = {
        "USDT": USDT_BUDGET.side_budget * Decimal("2") + USDT_BUDGET.reserve_quote,
        "FDUSD": FDUSD_BUDGET.side_budget * Decimal("2") + FDUSD_BUDGET.reserve_quote,
        "BTC": Decimal("0"),
        "ETH": Decimal("0"),
    }
    for portfolio in PORTFOLIOS.values():
        budget = budget_for_quote(portfolio.quote_asset)
        for pair in portfolio.pairs:
            base = pair.split("-")[0]
            requirements[base] += budget.side_budget / prices[pair]
    return requirements


def build_fdusd_bootstrap_plan(prices: Mapping[str, Decimal]) -> Dict[str, Any]:
    portfolio = PORTFOLIOS["FDUSD"]
    budget = FDUSD_BUDGET
    purchases = {}
    for pair in portfolio.pairs:
        price = Decimal(str(prices[pair]))
        if price <= 0:
            raise ValueError(f"Invalid bootstrap price for {pair}.")
        purchases[pair] = {
            "quote_amount": str(budget.side_budget),
            "estimated_base_amount": str(budget.side_budget / price),
            "maximum_slippage_pct": "0.1",
        }
    return {
        "quote_asset": "FDUSD",
        "minimum_balance": str(budget.capital_limit),
        "recommended_balance": str(FDUSD_RECOMMENDED_BALANCE),
        "strategy_budget": str(budget.strategy_budget),
        "strategy_reserve": str(budget.reserve_quote),
        "external_safety_buffer": str(FDUSD_RECOMMENDED_BALANCE - budget.capital_limit),
        "purchases": purchases,
        "expected_remaining_strategy_fdusd": str(
            budget.capital_limit - budget.side_budget * Decimal(len(portfolio.pairs))
        ),
        "automatic_rollback": False,
    }


def selection_grid_levels(half_range: Decimal, minimum_spread: Decimal) -> int:
    per_side = max(
        2,
        min(
            int(FDUSD_BUDGET.side_budget / MIN_ORDER_QUOTE),
            int(half_range / minimum_spread),
        ),
    )
    return per_side * 2


def validate_active_selection(payload: Mapping[str, Any], maker_rate: Decimal | None = None) -> Dict[str, Any]:
    schema_version = int(payload.get("schema_version", -1))
    if schema_version not in SUPPORTED_ACTIVE_SELECTION_SCHEMA_VERSIONS:
        raise ValueError("Unsupported active selection schema version.")
    if not str(payload.get("parameter_version", "")).strip():
        raise ValueError("Active selection requires a parameter_version.")
    if tuple(payload.get("trading_pairs", ())) != PORTFOLIOS["FDUSD"].pairs:
        raise ValueError("Active selection must target BTC-FDUSD and ETH-FDUSD.")
    if schema_version == 2:
        raw_pairs = payload.get("pair_parameters")
        if not isinstance(raw_pairs, Mapping) or set(raw_pairs) != set(APPROVED_PAIR_PROFILES):
            raise ValueError("Schema-v2 selection requires exactly BTC-FDUSD and ETH-FDUSD pair_parameters.")
        pair_parameters: Dict[str, Dict[str, Any]] = {}
        for pair, required_profile in APPROVED_PAIR_PROFILES.items():
            raw_pair = raw_pairs[pair]
            if not isinstance(raw_pair, Mapping):
                raise ValueError(f"{pair} parameters are missing.")
            profile = str(raw_pair.get("profile", ""))
            if profile != required_profile:
                raise ValueError(f"{pair} must use approved profile {required_profile}.")
            approved = BINANCE_AI_GRID_PROFILES[profile]
            candidate = {
                "grid_range": Decimal(str(raw_pair["grid_range"])),
                "grid_levels": int(raw_pair["grid_levels"]),
                "take_profit": Decimal(str(raw_pair["take_profit"])),
                "minimum_order_quote": Decimal(str(raw_pair["minimum_order_quote"])),
                "move_threshold": Decimal(str(raw_pair["move_threshold"])),
                "min_grid_move_seconds": int(raw_pair["min_grid_move_seconds"]),
                "order_refresh_seconds": int(raw_pair["order_refresh_seconds"]),
            }
            if candidate != approved:
                raise ValueError(f"{pair} parameters do not match immutable profile {profile}.")
            if candidate["grid_levels"] < 4 or candidate["grid_levels"] % 2:
                raise ValueError(f"{pair} grid_levels must be an even number of at least four.")
            if candidate["minimum_order_quote"] < Decimal("10"):
                raise ValueError(f"{pair} minimum_order_quote must be at least 10 FDUSD.")
            pair_parameters[pair] = {"profile": profile, **candidate}
        return {
            "schema_version": schema_version,
            "parameter_version": str(payload["parameter_version"]),
            "selection_sha256": hashlib.sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":"), default=str,
            ).encode("utf-8")).hexdigest(),
            "pair_parameters": pair_parameters,
        }

    raw = payload.get("parameters")
    if not isinstance(raw, Mapping):
        raise ValueError("Active selection parameters are missing.")
    half_range = Decimal(str(raw["half_range"]))
    minimum_spread = Decimal(str(raw["minimum_spread"]))
    configured_tp = Decimal(str(raw["take_profit"]))
    move_threshold = Decimal(str(raw["move_threshold"]))
    move_cooldown = int(raw.get("min_grid_move_seconds", GRID_MOVE_COOLDOWN_SECONDS))
    if half_range not in ALLOWED_HALF_RANGES:
        raise ValueError("half_range is outside the approved search space.")
    if minimum_spread not in ALLOWED_MIN_SPREADS:
        raise ValueError("minimum_spread is outside the approved search space.")
    if configured_tp not in ALLOWED_TAKE_PROFITS:
        raise ValueError("take_profit is outside the approved search space.")
    if move_threshold not in ALLOWED_MOVE_THRESHOLDS:
        raise ValueError("move_threshold is outside the approved search space.")
    if move_cooldown != GRID_MOVE_COOLDOWN_SECONDS:
        raise ValueError("Grid move cooldown must remain 30 minutes.")
    effective_tp = effective_take_profit(maker_rate or Decimal("0"), configured_tp)
    legacy = {
        "grid_range": half_range * Decimal("2"),
        "grid_levels": selection_grid_levels(half_range, minimum_spread),
        "minimum_spread": minimum_spread,
        "take_profit": effective_tp,
        "move_threshold": move_threshold,
        "min_grid_move_seconds": move_cooldown,
        "minimum_order_quote": MIN_ORDER_QUOTE,
        "order_refresh_seconds": ORDER_REFRESH_SECONDS,
    }
    return {
        **legacy,
        "schema_version": schema_version,
        "parameter_version": str(payload["parameter_version"]),
        "selection_sha256": hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")).hexdigest(),
        "pair_parameters": {
            pair: {"profile": "legacy_shared", **legacy}
            for pair in PORTFOLIOS["FDUSD"].pairs
        },
    }


def build_live_config(portfolio: GridPortfolio, prices: Mapping[str, Decimal], maker_rate: Decimal,
                      trading_enabled: bool = False,
                      reserved_base_by_pair: Mapping[str, Decimal] | None = None,
                      bootstrap_from_quote: bool = False,
                      bootstrap_completed: bool = False) -> Dict[str, Any]:
    take_profit = effective_take_profit(maker_rate)
    budget = budget_for_quote(portfolio.quote_asset)
    base_reservations = reserved_base_by_pair or {
        pair: budget.side_budget / prices[pair] for pair in portfolio.pairs
    }
    reservations = {pair: str(base_reservations[pair]) for pair in portfolio.pairs}
    return {
        "script_file_name": "walk_forward_portfolio_grid_live.py",
        "controllers_config": [],
        "exchange": CONNECTOR,
        "trading_pairs": list(portfolio.pairs),
        "quote_asset": portfolio.quote_asset,
        "capital_limit_quote": float(budget.capital_limit),
        "strategy_budget_quote": float(budget.strategy_budget),
        "reserve_quote": float(budget.reserve_quote),
        "pair_budget_quote": float(budget.pair_budget),
        "side_budget_quote": float(budget.side_budget),
        "reserved_base_by_pair": reservations,
        "grid_range": 0.08,
        "grid_levels": 8,
        "take_profit": float(take_profit),
        "move_threshold": 0.02,
        "min_grid_move_seconds": 1800,
        "order_refresh_time": ORDER_REFRESH_SECONDS,
        "pair_breakers_enabled": True,
        "pair_loss_breaker_enabled": _risk_enabled("GRID_RISK_STRATEGY_LOSS_BREAKER_ENABLED"),
        "pair_drawdown_breaker_enabled": _risk_enabled("GRID_RISK_STRATEGY_DRAWDOWN_BREAKER_ENABLED"),
        "portfolio_breakers_enabled": True,
        "portfolio_loss_breaker_enabled": _risk_enabled("GRID_RISK_PORTFOLIO_LOSS_BREAKER_ENABLED"),
        "portfolio_drawdown_breaker_enabled": _risk_enabled("GRID_RISK_PORTFOLIO_DRAWDOWN_BREAKER_ENABLED"),
        "cost_floor_enabled": _risk_enabled("GRID_RISK_POSITION_PROTECTION_ENABLED"),
        "inventory_exit_enabled": _risk_enabled("GRID_RISK_POSITION_PROTECTION_ENABLED"),
        "max_extra_inventory_quote": 10,
        "profit_protection_seconds": 86400,
        "max_extra_inventory_hold_seconds": 172800,
        "risk_state_persist_seconds": RISK_STATE_PERSIST_SECONDS,
        "startup_order_reconcile_seconds": STARTUP_ORDER_RECONCILE_SECONDS,
        "portfolio_stop_loss_quote": float(budget.portfolio_loss_limit),
        "pair_stop_loss_quote": float(budget.pair_loss_limit),
        "portfolio_drawdown_limit_pct": float(PORTFOLIO_DRAWDOWN_LIMIT_PCT),
        "pair_drawdown_limit_pct": float(PAIR_DRAWDOWN_LIMIT_PCT),
        "fail_closed_seconds": 60,
        "min_order_quote": float(MIN_ORDER_QUOTE),
        "fee_rate": float(maker_rate),
        "trading_enabled": trading_enabled,
        "bootstrap_from_quote": bootstrap_from_quote,
        "bootstrap_completed": bootstrap_completed,
        "active_selection_file": "data/active_selection.json",
        "runtime_state_file": "data/live_grid_runtime_state.json",
        "parameter_poll_seconds": 60,
        "active_parameter_version": "bootstrap-static-v1",
        "macro_gate_enabled": portfolio.quote_asset == "FDUSD" and _risk_enabled("GRID_RISK_FOMC_GATE_ENABLED"),
        "macro_gate_file": "data/macro_gate.json",
        "macro_gate_poll_seconds": 5,
        "macro_gate_max_age_seconds": 150,
        "macro_fail_closed": True,
        "technical_buy_gate_enabled": portfolio.quote_asset == "FDUSD" and _risk_enabled("GRID_RISK_V22_WEEKLY_GATE_ENABLED"),
        "technical_buy_gate_file": "data/xgboost_risk_gate.json",
        "technical_buy_gate_poll_seconds": 5,
        "technical_buy_gate_max_age_seconds": 150,
        "technical_buy_fail_closed": True,
        "technical_model_sha256": "",
        "technical_feature_sha256": "",
    }


def validate_live_config(config: Mapping[str, Any]) -> None:
    quote = str(config.get("quote_asset", "")).upper()
    if config.get("exchange") != CONNECTOR or "perpetual" in str(config.get("exchange", "")):
        raise ValueError("Live grid must use the Binance spot connector.")
    if quote not in PORTFOLIOS:
        raise ValueError("Only USDT and FDUSD live portfolios are supported.")
    if tuple(config.get("trading_pairs", ())) != PORTFOLIOS[quote].pairs:
        raise ValueError("Live grid pairs must be BTC and ETH for the selected quote asset.")
    budget = budget_for_quote(quote)
    if Decimal(str(config.get("capital_limit_quote"))) != budget.capital_limit:
        raise ValueError(f"Capital limit must be exactly {budget.capital_limit} {quote}.")
    if Decimal(str(config.get("strategy_budget_quote"))) != budget.strategy_budget:
        raise ValueError(f"Strategy budget must be exactly {budget.strategy_budget} {quote}.")
    if Decimal(str(config.get("reserve_quote"))) != budget.reserve_quote:
        raise ValueError(f"Reserve must be exactly {budget.reserve_quote} {quote}.")
    if Decimal(str(config.get("pair_budget_quote"))) != budget.pair_budget:
        raise ValueError("Pair allocation does not match the approved budget.")
    if Decimal(str(config.get("side_budget_quote"))) != budget.side_budget:
        raise ValueError("Side allocation does not match the approved budget.")
    if Decimal(str(config.get("portfolio_stop_loss_quote"))) != budget.portfolio_loss_limit:
        raise ValueError("Portfolio stop loss does not match the approved risk limit.")
    if Decimal(str(config.get("pair_stop_loss_quote"))) != budget.pair_loss_limit:
        raise ValueError("Pair stop loss does not match the approved risk limit.")
    if Decimal(str(config.get("portfolio_drawdown_limit_pct"))) != PORTFOLIO_DRAWDOWN_LIMIT_PCT:
        raise ValueError("Portfolio peak drawdown limit must be exactly 6%.")
    if Decimal(str(config.get("pair_drawdown_limit_pct"))) != PAIR_DRAWDOWN_LIMIT_PCT:
        raise ValueError("Pair peak drawdown limit must be exactly 3%.")
    if int(config.get("order_refresh_time", 0)) != ORDER_REFRESH_SECONDS:
        raise ValueError("Live Grid order refresh time must be exactly 2 hours.")
    if bool(config.get("inventory_exit_enabled", True)):
        if Decimal(str(config.get("max_extra_inventory_quote"))) != Decimal("10"):
            raise ValueError("Live Grid extra inventory cap must be exactly 10 quote units.")
        if int(config.get("profit_protection_seconds", 0)) != 86400:
            raise ValueError("Live Grid profit protection must last exactly 24 hours.")
        if int(config.get("max_extra_inventory_hold_seconds", 0)) != 172800:
            raise ValueError("Live Grid extra inventory must Taker-exit after exactly 48 hours.")
    if int(config.get("risk_state_persist_seconds", 0)) != RISK_STATE_PERSIST_SECONDS:
        raise ValueError("Live Grid risk state must be persisted every 5 seconds.")
    if int(config.get("startup_order_reconcile_seconds", 0)) != STARTUP_ORDER_RECONCILE_SECONDS:
        raise ValueError("Live Grid startup order reconciliation must last exactly 30 seconds.")
    reservations = config.get("reserved_base_by_pair", {})
    if set(reservations) != set(PORTFOLIOS[quote].pairs):
        raise ValueError("Every live pair requires an explicit base reservation.")
    if any(Decimal(str(value)) <= 0 for value in reservations.values()):
        raise ValueError("Base reservations must be positive.")
    if bool(config.get("trading_enabled")) and bool(config.get("bootstrap_from_quote")):
        if not bool(config.get("bootstrap_completed")):
            raise ValueError("Quote-only bootstrap must complete before live trading is enabled.")
    if quote == "FDUSD":
        if bool(config.get("macro_gate_enabled")) and not bool(config.get("macro_fail_closed")):
            raise ValueError("FDUSD FOMC macro gate must fail closed.")
        if not 30 <= int(config.get("macro_gate_max_age_seconds", 0)) <= 180:
            raise ValueError("FDUSD FOMC macro gate freshness must be between 30 and 180 seconds.")
        if not 1 <= int(config.get("macro_gate_poll_seconds", 0)) <= 30:
            raise ValueError("FDUSD FOMC macro gate polling must be between 1 and 30 seconds.")
        if bool(config.get("technical_buy_gate_enabled")) and Path(str(config.get("technical_buy_gate_file", ""))).name != "xgboost_risk_gate.json":
            raise ValueError("FDUSD live Grid does not permit a Mechanism 1 technical-gate file.")
        if bool(config.get("trading_enabled")) and bool(config.get("technical_buy_gate_enabled")):
            hashes = (
                str(config.get("technical_model_sha256", "")),
                str(config.get("technical_feature_sha256", "")),
            )
            if any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value.lower())
                for value in hashes
            ):
                raise ValueError("Enabled FDUSD Grid requires locked XGBoost model and feature hashes.")
        if bool(config.get("technical_buy_gate_enabled")) and not bool(config.get("technical_buy_fail_closed")):
            raise ValueError("FDUSD XGBoost BUY gate must fail closed.")
        if not 30 <= int(config.get("technical_buy_gate_max_age_seconds", 0)) <= 180:
            raise ValueError("FDUSD technical BUY gate freshness must be between 30 and 180 seconds.")
        if not 1 <= int(config.get("technical_buy_gate_poll_seconds", 0)) <= 30:
            raise ValueError("FDUSD technical BUY gate polling must be between 1 and 30 seconds.")


def validate_exchange_filters(symbol_info: Mapping[str, Any], order_quote: Decimal) -> None:
    if symbol_info.get("status") != "TRADING":
        raise ValueError(f"Symbol {symbol_info.get('symbol')} is not trading.")
    filters = {item["filterType"]: item for item in symbol_info.get("filters", [])}
    notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL")
    if notional is None:
        raise ValueError("Missing Binance notional filter.")
    minimum = Decimal(str(notional.get("minNotional", notional.get("notional"))))
    if order_quote < minimum * Decimal("1.05"):
        raise ValueError(f"Order {order_quote} is too close to minimum notional {minimum}.")
    if "LOT_SIZE" not in filters or "PRICE_FILTER" not in filters:
        raise ValueError("Missing Binance quantity or price filter.")


def extract_balances(payload: Any) -> Dict[str, Decimal]:
    """Normalize legacy balance maps and current portfolio token-row responses.

    Deployment checks intentionally use ``available_units`` before total units so
    locked funds cannot be counted as spendable Grid capital.
    """
    balances: Dict[str, Decimal] = {}

    def add(asset: Any, raw: Any) -> None:
        if asset is None:
            return
        try:
            key = str(asset).upper()
            balances[key] = balances.get(key, Decimal("0")) + Decimal(str(raw))
        except Exception:
            return

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            token = node.get("token") or node.get("asset") or node.get("symbol")
            if token is not None and any(
                field in node for field in ("available_units", "units", "total", "balance")
            ):
                add(token, node.get(
                    "available_units", node.get("units", node.get("total", node.get("balance", 0)))
                ))
                return
            if "balances" in node:
                visit(node["balances"])
                return
            for asset, raw in node.items():
                if isinstance(raw, Mapping) and any(
                    field in raw for field in ("available_units", "units", "total", "balance")
                ):
                    add(asset, raw.get(
                        "available_units", raw.get("units", raw.get("total", raw.get("balance", 0)))
                    ))
                elif isinstance(raw, (Mapping, Sequence)) and not isinstance(raw, (str, bytes)):
                    visit(raw)
                elif str(asset).upper() == str(asset):
                    add(asset, raw)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for item in node:
                visit(item)

    visit(payload)
    return balances
