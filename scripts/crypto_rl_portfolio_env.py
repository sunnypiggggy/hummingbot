"""Isolated, cash-funded daily crypto portfolio environment.

Hourly labels are hour opens. Decisions use only prior completed daily bars.
At 00:00 UTC reduce/sell, at 01:00 UTC buy increases; mark every hour close.
The action is a target portfolio, never a permission to borrow or short.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    _BaseEnv = gym.Env
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

    class _BaseEnv:
        def reset(self, *, seed=None, options=None):
            if seed is not None or not hasattr(self, "np_random"):
                self.np_random = np.random.default_rng(seed)

    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low, self.high, self.shape, self.dtype = low, high, shape, np.dtype(dtype)
            self._rng = np.random.default_rng()

        def seed(self, seed=None):
            self._rng = np.random.default_rng(seed)

        def sample(self):
            return self._rng.uniform(self.low, self.high, self.shape).astype(self.dtype)

        def contains(self, value):
            a = np.asarray(value)
            return a.shape == self.shape and np.isfinite(a).all() and (a >= self.low).all() and (a <= self.high).all()

    class spaces:
        Box = _Box


def utc(value):
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def valid_price(value):
    return np.isfinite(value) and value > 0


class PortfolioLedger:
    """Fractional-unit cash/coin ledger, keyed by immutable asset identity.

    Fee is paid in USDT on actual adverse fill notional. Weighted average cost
    is kept only to reconcile realized/open PNL; it never determines valuation.
    No historical lot-size or minimum-notional reconstruction is claimed.
    """
    def __init__(self, capital=10000., fee=.001, slippage=.0002, assets=None):
        if not np.isfinite(capital) or capital <= 0:
            raise ValueError("capital must be finite and positive")
        if not (0 <= fee < 1 and 0 <= slippage < 1):
            raise ValueError("fee/slippage must be fractions in [0,1)")
        self.initial_capital = self.cash = float(capital)
        self.fee, self.slippage = float(fee), float(slippage)
        self.assets = assets or {}
        self.positions: dict[str, float] = {}
        self.cost_basis: dict[str, float] = {}
        self.realized_pnl = 0.
        self.fills: list[dict[str, Any]] = []

    def value(self, prices):
        total = self.cash
        for key, quantity in self.positions.items():
            price = float(prices.get(key, np.nan))
            if not valid_price(price):
                raise ValueError(f"Held asset has no valid mark price: {key}")
            total += quantity * price
        return float(total)

    def _fill(self, key, quantity, reference, timestamp, side, reason):
        key = str(key)
        quantity, reference = float(quantity), float(reference)
        if not np.isfinite(quantity) or quantity < 0 or not valid_price(reference):
            raise ValueError("Invalid quantity or execution reference")
        if quantity == 0:
            return None
        direction = 1 if side == "BUY" else -1
        price = reference * (1 + direction * self.slippage)
        notional = quantity * price
        fee = notional * self.fee
        realized = 0.
        if side == "BUY":
            charge = notional + fee
            if charge > self.cash + 1e-8:
                raise ValueError("Insufficient cash including buy fee")
            self.cash -= charge
            self.positions[key] = self.positions.get(key, 0.) + quantity
            self.cost_basis[key] = self.cost_basis.get(key, 0.) + charge
        else:
            owned = self.positions.get(key, 0.)
            if quantity > owned + 1e-12 or owned <= 0:
                raise ValueError("Cannot sell more than strategy-owned inventory")
            quantity = min(quantity, owned)
            notional, fee = quantity * price, quantity * price * self.fee
            removed_basis = self.cost_basis[key] * quantity / owned
            self.cash += notional - fee
            realized = notional - fee - removed_basis
            self.realized_pnl += realized
            remainder = owned - quantity
            if remainder <= 1e-14:
                del self.positions[key]
                del self.cost_basis[key]
            else:
                self.positions[key] = remainder
                self.cost_basis[key] -= removed_basis
        if -1e-8 < self.cash < 0:
            self.cash = 0.
        if self.cash < 0 or any(q < 0 or not np.isfinite(q) for q in self.positions.values()):
            raise AssertionError("Invalid cash/coin ledger")
        fill = {
            "timestamp": utc(timestamp), "asset_key": key,
            "symbol": self.assets.get(key, {}).get("symbol", key),
            "side": side, "qty": quantity, "reference_price": reference,
            "price": price, "notional": notional, "fee": fee,
            "slippage_cost": quantity * abs(price - reference), "reason": reason,
            "cash_after": self.cash, "quantity_after": self.positions.get(key, 0.),
            "realized_pnl": realized,
        }
        self.fills.append(fill)
        return fill

    def buy(self, key, quantity, reference, timestamp, reason="target_increase"):
        return self._fill(key, quantity, reference, timestamp, "BUY", reason)

    def sell(self, key, quantity, reference, timestamp, reason="target_reduction"):
        return self._fill(key, quantity, reference, timestamp, "SELL", reason)

    def buy_with_cash(self, key, budget, reference, timestamp, reason="initial_purchase"):
        budget = float(budget)
        if budget < 0 or budget > self.cash + 1e-8:
            raise ValueError("Cash budget outside available funds")
        quantity = budget / (reference * (1 + self.slippage) * (1 + self.fee))
        return self.buy(key, quantity, reference, timestamp, reason)


class CryptoPortfolioEnv(_BaseEnv):
    """Daily Gymnasium env with 10 dynamic rank slots and one cash bucket.

    action_to_weights maps Box[-1,1]^11 to a constrained 11-vector.
    step_target accepts that vector directly and advances exactly one day.
    step_hold is a passive-reference operation: no rebalancing at all.
    rebalance_target is an internal helper called once per daily step.
    """
    metadata = {"render_modes": []}
    FEATURE_NAMES = ("return_1d/.1", "return_7d/.3", "return_30d/.5",
                     "volatility_30d/.1", "sma30_deviation/.3",
                     "volume_relative30/3", "cap_share", "current_weight", "feature_valid")
    FILL_COLUMNS = ("timestamp", "asset_key", "symbol", "side", "qty", "reference_price",
                    "price", "notional", "fee", "slippage_cost", "reason", "cash_after",
                    "quantity_after", "realized_pnl")

    def __init__(self, dataset, start, end, costs=None, capital=10000.,
                 training=False, episode_days=128, cash_floor=.05,
                 max_asset_weight=.25, constraints=True):
        super().__init__()
        self.dataset = dataset
        self.start, self.end = utc(start), utc(end)
        if self.start != self.start.normalize() or self.end != self.end.normalize() or self.start >= self.end:
            raise ValueError("Require nonempty UTC daily [start,end) boundaries")
        self.capital = float(capital)
        self.training, self.episode_days = bool(training), int(episode_days)
        if self.episode_days < 1:
            raise ValueError("episode_days must be positive")
        self.cash_floor = float(cash_floor)
        self.max_asset_weight = float(max_asset_weight) if constraints else 1.
        if not (0 <= self.cash_floor < 1 and 0 < self.max_asset_weight <= 1):
            raise ValueError("Invalid cash floor or asset cap")
        self.costs = dict(costs or {"fee": .001, "slippage": .0002})
        self.assets = {str(k): v for k, v in dataset.get("assets", dataset.get("registry", {})).items()}
        self._frames = {}
        reference_index = reference_columns = None
        for field in ("open", "high", "low", "close", "volume"):
            frame = dataset["hourly_" + field]
            if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
                raise ValueError("Market hours require timezone-aware index")
            frame = frame.copy(deep=False)
            frame.index = frame.index.tz_convert("UTC")
            frame.columns = frame.columns.map(str)
            if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
                raise ValueError("Market hour index must be unique and ordered")
            if (frame.index != frame.index.floor("h")).any():
                raise ValueError("Market hours must be aligned hour opens")
            if reference_index is None:
                reference_index, reference_columns = frame.index, frame.columns
            if not frame.index.equals(reference_index) or not frame.columns.equals(reference_columns):
                raise ValueError("All hourly frames must share timestamps and identities")
            self._frames[field] = frame.astype(float)
        expected = pd.date_range(reference_index[0], reference_index[-1], freq="h")
        if not reference_index.equals(expected):
            raise ValueError("Market hourly grid contains omitted rows")
        if self.start < reference_index[0] or self.end > reference_index[-1] + pd.Timedelta(hours=1):
            raise ValueError("Evaluation interval lies outside market data")
        self._index = reference_index
        self._hour_lookup = {t: i for i, t in enumerate(reference_index)}
        self._column_ids = list(reference_columns)
        self._column_lookup = {key: i for i, key in enumerate(self._column_ids)}
        self._arrays = {key: frame.to_numpy() for key, frame in self._frames.items()}
        universe = dataset["universe"].copy()
        universe["date"] = pd.to_datetime(universe["date"], utc=True)
        if "available_after" in universe:
            universe["available_after"] = pd.to_datetime(universe["available_after"], utc=True)
            if (universe.available_after > universe.date).any():
                raise ValueError("Universe contains a ranking snapshot unavailable at decision time")
        if (universe.date != universe.date.dt.normalize()).any():
            raise ValueError("Universe dates must be UTC midnight")
        self._universe = {}
        for day, rows in universe.groupby("date", sort=True):
            rows = rows.sort_values("slot", kind="stable").copy()
            if rows.slot.tolist() != list(range(10)):
                raise ValueError(f"Universe must contain unique slots 0..9: {day}")
            keys = [str(k) if pd.notna(k) and str(k) else None for k in rows.asset_key]
            existing = [key for key in keys if key is not None]
            if len(existing) != len(set(existing)):
                raise ValueError("Duplicate economic asset identity in universe")
            if any(key not in self._column_lookup for key in existing):
                raise ValueError("Universe asset lacks a corresponding market column")
            rows["asset_key"] = keys
            self._universe[day] = rows
        self._feature_cube, self._feature_valid = self._make_features()
        self._eligible_starts, self.training_window_metadata = self.train_complete_window_candidates()
        self.observation_space = spaces.Box(-10., 10., shape=(92,), dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., shape=(11,), dtype=np.float32)
        self._inside_step = False
        self._done = True
        self.ledger = None
        self.training_episode_history = []

    def _make_features(self):
        complete = None
        for field in ("open", "high", "low", "close", "volume"):
            frame = self._frames[field]
            finite = frame.notna() & np.isfinite(frame)
            finite &= frame.ge(0) if field == "volume" else frame.gt(0)
            complete = finite if complete is None else complete & finite
        count = complete.resample("D").sum()
        daily_close = self._frames["close"].resample("D").last().where(count.eq(24))
        daily_volume = self._frames["volume"].resample("D").sum(min_count=24).where(count.eq(24))
        returns = daily_close.pct_change(fill_method=None)
        mean_price = daily_close.rolling(30, min_periods=30).mean()
        mean_volume = daily_volume.rolling(30, min_periods=30).mean()
        pieces = [
            returns / .1, (daily_close / daily_close.shift(7) - 1) / .3,
            (daily_close / daily_close.shift(30) - 1) / .5,
            returns.rolling(30, min_periods=30).std(ddof=1) / .1,
            (daily_close / mean_price - 1) / .3,
            (daily_volume / mean_volume - 1) / 3.,
        ]
        valid = count.eq(24).rolling(31, min_periods=31).sum().eq(31) & mean_volume.gt(0)
        for piece in pieces:
            valid &= piece.notna() & np.isfinite(piece)
        days = daily_close.index + pd.Timedelta(days=1)
        cube = np.stack([p.to_numpy() for p in pieces], axis=2)
        mask = valid.to_numpy()
        cube = np.where(mask[:, :, None], np.clip(np.nan_to_num(cube), -10, 10), 0.)
        return {d: cube[i] for i, d in enumerate(days)}, {d: mask[i] for i, d in enumerate(days)}

    def _prices(self, timestamp, field):
        at = self._hour_lookup.get(utc(timestamp))
        if at is None:
            raise ValueError(f"No execution/valuation hour: {timestamp}")
        return dict(zip(self._column_ids, self._arrays[field][at]))

    def _rows_at(self, day):
        if day >= self.end:
            day = day - pd.Timedelta(days=1)
        if day not in self._universe:
            if day >= self.episode_end:
                day = day - pd.Timedelta(days=1)
            if day not in self._universe:
                raise ValueError(f"No historical universe for decision date: {day}")
        return self._universe[day]

    @property
    def current_universe(self):
        return self._rows_at(self.current_date).copy()

    @property
    def current_assets(self):
        return self.current_universe.asset_key.tolist()

    @property
    def current_valid_mask(self):
        available = self._feature_valid.get(self.current_date)
        return np.array([bool(key is not None and available is not None and available[self._column_lookup[key]]
                              and not self._known_delisting_today(key, self.current_date))
                         for key in self.current_assets], dtype=bool)

    def _known_delisting_today(self, key, day):
        asset = self.assets.get(key, {})
        notice = asset.get("delisting_notice_at")
        until = asset.get("effective_until")
        if notice is None or until is None:
            return False
        notice, until = utc(notice), utc(until)
        return bool(notice <= day and day < until <= day + pd.Timedelta(days=1))

    def train_complete_window_candidates(self):
        """Exclude only incomplete TRAIN episodes; never filter evaluation days."""
        if not self.training:
            return [], {"applied": False, "reason": "evaluation remains continuous and unfiltered"}
        length = (self.end - self.start).days
        episode = min(self.episode_days, length)
        dates = pd.date_range(self.start, self.end, freq="D", inclusive="left")
        open_ok = (np.isfinite(self._frames["open"]) & self._frames["open"].gt(0)).resample("D").sum().eq(24)
        close_ok = (np.isfinite(self._frames["close"]) & self._frames["close"].gt(0)).resample("D").sum().eq(24)
        bad = {}
        day_keys = {}
        for day in dates:
            issues = []
            rows = self._universe.get(day)
            if rows is None:
                bad[day] = ["missing_universe"]
                day_keys[day] = []
                continue
            keys = [key for key in rows.asset_key if key is not None]
            active = [key for key in keys if not self._known_delisting_today(key, day)]
            day_keys[day] = active
            for key in active:
                if day not in open_ok.index or not (open_ok.loc[day, key] and close_ok.loc[day, key]):
                    issues.append("incomplete_24h_open_close:" + key)
            previous = self._universe.get(day - pd.Timedelta(days=1))
            previous_keys = [] if previous is None else [
                key for key in previous.asset_key
                if key is not None and not self._known_delisting_today(key, day - pd.Timedelta(days=1))]
            exiting = set(previous_keys) - set(active)
            for key in exiting:
                for hour in (0, 1):
                    if not valid_price(self._prices(day + pd.Timedelta(hours=hour), "open").get(key, np.nan)):
                        issues.append("outgoing_execution_quote:" + key + ":" + str(hour))
            if issues:
                bad[day] = issues
        candidates = []
        exclusions = []
        covered = set()
        for first in range(length - episode + 1):
            selected = dates[first:first + episode]
            start, end = selected[0], selected[-1] + pd.Timedelta(days=1)
            issues = [str(day.date()) for day in selected if day in bad]
            if not issues and end < self.end:
                prices = self._prices(end, "open")
                if any(not valid_price(prices.get(key, np.nan)) for key in day_keys[selected[-1]]):
                    issues = ["missing_bootstrap_open:" + str(end)]
            if issues:
                exclusions.append({"start": str(start), "end": str(end), "excluded_dates": issues})
            else:
                candidates.append(start)
                covered.update(selected)
        information = {
            "applied": True, "selection_basis": "TRAIN market completeness only, never PNL",
            "requested_train_days": length, "episode_days": episode,
            "candidate_windows": length - episode + 1, "eligible_windows": len(candidates),
            "excluded_windows": len(exclusions), "covered_training_days": len(covered),
            "incomplete_days": [{"date": str(day), "reasons": reasons} for day, reasons in bad.items()],
            "excluded_window_details": exclusions,
        }
        if not candidates:
            raise ValueError("No complete training episodes remain; do not interpolate market gaps")
        return candidates, information

    def _observation(self):
        rows = self.current_universe
        prices = self._prices(self.current_date - pd.Timedelta(hours=1), "close")
        nav = self.ledger.value(prices)
        keys = rows.asset_key.tolist()
        cube = self._feature_cube.get(self.current_date)
        cap_column = "cap" if "cap" in rows else "market_cap"
        caps = pd.to_numeric(rows[cap_column], errors="coerce").fillna(0).clip(lower=0).to_numpy()
        denominator = caps.sum()
        features = np.zeros((10, 9), dtype=float)
        mask = self.current_valid_mask
        for i, key in enumerate(keys):
            if key is None:
                continue
            col = self._column_lookup[key]
            if cube is not None:
                features[i, :6] = cube[col]
            features[i, 6] = caps[i] / denominator if denominator else 0.
            quantity = self.ledger.positions.get(key, 0.)
            features[i, 7] = quantity * prices[key] / nav if quantity else 0.
            features[i, 8] = mask[i]
        outside = sum(q * prices[key] for key, q in self.ledger.positions.items() if key not in keys) / nav
        obs = np.r_[features.reshape(-1), self.ledger.cash / nav, outside]
        if not np.isfinite(obs).all():
            raise ValueError("Observation contains non-finite values")
        return np.clip(obs, -10, 10).astype(np.float32)

    def action_to_weights(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (11,) or not np.isfinite(action).all():
            raise ValueError("Action must be eleven finite numbers")
        scores = np.clip((action + 1) / 2, 0, 1)
        scores[:10] *= self.current_valid_mask
        total = scores.sum()
        result = np.zeros(11)
        if total > 0:
            result[:10] = np.minimum((1 - self.cash_floor) * scores[:10] / total, self.max_asset_weight)
        result[10] = 1 - result[:10].sum()
        return result

    def _validate_target(self, weights):
        result = np.asarray(weights, dtype=float).copy()
        if result.shape != (11,) or not np.isfinite(result).all() or (result < -1e-12).any():
            raise ValueError("Target must be eleven finite nonnegative weights")
        if abs(result.sum() - 1) > 1e-8 or result[10] < self.cash_floor - 1e-10:
            raise ValueError("Target weights must sum to one and respect cash floor")
        if (result[:10] > self.max_asset_weight + 1e-10).any():
            raise ValueError("Target exceeds asset concentration cap")
        result[:10] *= self.current_valid_mask
        result[10] = 1 - result[:10].sum()
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        length = (self.end - self.start).days
        if self.training:
            self.episode_start = self._eligible_starts[int(self.np_random.integers(0, len(self._eligible_starts)))]
            self.episode_end = self.episode_start + pd.Timedelta(days=min(length, self.episode_days))
        else:
            self.episode_start, self.episode_end = self.start, self.end
        self.current_date = self.episode_start
        if self.training:
            self.training_episode_history.append({
                "start": str(self.episode_start), "end_exclusive": str(self.episode_end),
                "completed_until": str(self.episode_start), "completed_steps": 0,
                "terminated": False, "truncated": False,
            })
        self.ledger = PortfolioLedger(self.capital, **self.costs, assets=self.assets)
        self._done = False
        self._equity_rows, self._weight_rows = [], []
        self._boundary_rows = [{"timestamp": self.current_date, "equity": self.capital, "reward": 0.}]
        self._nav = self.capital
        self._last_prices = None
        return self._observation(), {"date": self.current_date, "equity": self._nav}

    @staticmethod
    def _solve_nav(initial, cost, volume_function):
        if cost == 0 or volume_function(initial) <= 1e-12:
            return initial
        low, high = 0., initial
        for _ in range(55):
            middle = (low + high) / 2
            if middle + cost * volume_function(middle) > initial:
                high = middle
            else:
                low = middle
        return (low + high) / 2

    def rebalance_target(self, weights):
        if not self._inside_step:
            raise RuntimeError("Use step_target once; rebalance_target is its internal execution helper")
        day = self.current_date
        keys = self.current_assets
        targets = {key: float(weights[i]) for i, key in enumerate(keys) if key is not None}
        p0 = self._prices(day, "open")
        initial = self.ledger.value(p0)
        current = {key: q * p0[key] for key, q in self.ledger.positions.items()}
        sell_volume = lambda nav: sum(max(v - targets.get(key, 0.) * nav, 0.) for key, v in current.items())
        sell_cost = 1 - (1 - self.ledger.slippage) * (1 - self.ledger.fee)
        after_sell_nav = self._solve_nav(initial, sell_cost, sell_volume)
        for key, value in current.items():
            reduction = max(value - targets.get(key, 0.) * after_sell_nav, 0.)
            if reduction > 1e-9:
                # Multiplication followed by division can round one ULP above
                # owned units for very low-price coins. Sizing is bounded by
                # actual strategy inventory; the ledger's strict guard remains.
                quantity = min(reduction / p0[key], self.ledger.positions[key])
                self.ledger.sell(key, quantity, p0[key], day,
                                 "known_delisting_protection" if self._known_delisting_today(key, day)
                                 else ("universe_exit" if key not in targets else "target_reduction"))
        self._append_mark(day, self._prices(day, "close"))
        purchase_time = day + pd.Timedelta(hours=1)
        p1 = self._prices(purchase_time, "open")
        initial = self.ledger.value(p1)
        held = {key: q * p1[key] for key, q in self.ledger.positions.items()}
        for key, weight in targets.items():
            if weight > 0 and not valid_price(p1.get(key, np.nan)):
                raise ValueError(f"Eligible asset has no buy execution price: {key}")
        buy_volume = lambda nav: sum(max(weight * nav - held.get(key, 0.), 0.) for key, weight in targets.items() if weight > 0)
        buy_cost = (1 + self.ledger.slippage) * (1 + self.ledger.fee) - 1
        after_buy_nav = self._solve_nav(initial, buy_cost, buy_volume)
        desired = {key: max(weight * after_buy_nav - held.get(key, 0.), 0.)
                   for key, weight in targets.items() if weight > 0}
        total = sum(desired.values())
        cash_target = max(self.cash_floor, weights[10])
        affordable = max(0., (self.ledger.cash - cash_target * initial) / (1 + buy_cost - cash_target * buy_cost))
        scale = min(1., affordable / total) if total > 0 else 0.
        for key, mid_value in desired.items():
            if mid_value * scale > 1e-9:
                self.ledger.buy(key, mid_value * scale / p1[key], p1[key], purchase_time)

    def _append_mark(self, timestamp, prices):
        nav = self.ledger.value(prices)
        self._equity_rows.append({"timestamp": timestamp, "equity": nav, "cash": self.ledger.cash,
                                  "base_value": nav - self.ledger.cash,
                                  "exposure": (nav - self.ledger.cash) / nav})
        self._last_prices = prices

    def step(self, action):
        return self.step_target(self.action_to_weights(action))

    def step_target(self, weights):
        if self._done:
            raise RuntimeError("Episode ended; reset before another step")
        target = self._validate_target(weights) if weights is not None else None
        day, starting_nav = self.current_date, self._nav
        self._inside_step = True
        try:
            if target is not None:
                self.rebalance_target(target)
                first_remaining_hour = 1
            else:
                first_remaining_hour = 0
            for hour in range(first_remaining_hour, 24):
                time = day + pd.Timedelta(hours=hour)
                self._append_mark(time, self._prices(time, "close"))
        finally:
            self._inside_step = False
        final = day + pd.Timedelta(days=1)
        full_boundary = final >= self.end
        final_nav = self.ledger.value(self._last_prices if full_boundary else self._prices(final, "open"))
        if final_nav <= 0:
            raise ValueError("Nonpositive portfolio equity")
        reward = float(100 * np.log(final_nav / starting_nav))
        self._boundary_rows.append({"timestamp": final, "equity": final_nav, "reward": reward})
        current_prices = self._last_prices
        marked_nav = self.ledger.value(current_prices)
        for i, row in enumerate(self.current_universe.itertuples(index=False)):
            key = row.asset_key
            quantity = self.ledger.positions.get(key, 0.) if key is not None else 0.
            actual = quantity * current_prices[key] / marked_nav if quantity else 0.
            self._weight_rows.append({"date": day, "slot": i, "asset_key": key,
                                     "symbol": getattr(row, "symbol", key), "target_weight": target[i] if target is not None else None,
                                     "actual_weight": actual})
        self.current_date, self._nav = final, final_nav
        self._done = final >= self.episode_end
        truncated = self._done and self.episode_end < self.end
        terminated = self._done and not truncated
        if self.training:
            self.training_episode_history[-1].update(
                completed_until=str(final), completed_steps=(final - self.episode_start).days,
                terminated=bool(terminated), truncated=bool(truncated))
        return self._observation(), reward, bool(terminated), bool(truncated), {
            "date": final, "equity": final_nav, "cash": self.ledger.cash,
            "target_weights": target.copy() if target is not None else None,
        }

    def step_hold(self):
        return self.step_target(None)

    def get_result(self):
        if not self._equity_rows:
            raise RuntimeError("No completed daily steps")
        equity = pd.DataFrame(self._equity_rows).set_index("timestamp")
        equity.index.name = "timestamp"
        prices = self._last_prices
        final = self.ledger.value(prices)
        fills = pd.DataFrame(self.ledger.fills, columns=self.FILL_COLUMNS)
        base = final - self.ledger.cash
        fees = float(fills.fee.sum()) if len(fills) else 0.
        notional = float(fills.notional.sum()) if len(fills) else 0.
        slippage = float(fills.slippage_cost.sum()) if len(fills) else 0.
        open_pnl = sum(q * prices[key] - self.ledger.cost_basis[key] for key, q in self.ledger.positions.items())
        if abs(final - self.capital - self.ledger.realized_pnl - open_pnl) > 1e-6:
            raise AssertionError("Realized/open PNL bridge failed")
        holdings = [{"asset_key": key, "symbol": self.assets.get(key, {}).get("symbol", key),
                     "qty": q, "mark_price": prices[key], "market_value": q * prices[key],
                     "weight": q * prices[key] / final} for key, q in self.ledger.positions.items()]
        return {
            "equity": equity, "fills": fills, "weights": pd.DataFrame(self._weight_rows),
            "boundary_equity": pd.DataFrame(self._boundary_rows).set_index("timestamp"),
            "metadata": {
                "initial_capital": self.capital, "final_equity": final, "final_cash": self.ledger.cash,
                "final_positions": dict(self.ledger.positions), "final_holdings": holdings,
                "fees": fees, "total_notional": notional, "slippage_cost": slippage,
                "realized_pnl": self.ledger.realized_pnl, "unrealized_pnl": open_pnl,
                "final_liquidation_equity": self.ledger.cash + base * (1 - self.ledger.slippage) * (1 - self.ledger.fee),
                "fee_rate": self.ledger.fee, "slippage": self.ledger.slippage,
                "cash_floor_target": self.cash_floor, "max_asset_weight_target": self.max_asset_weight,
                "valuation_frequency": "1h", "equity_timestamp_semantics": "hour_open label, marked at hour_close",
                "execution_timing": "daily 00UTC reductions, 01UTC purchases; each leg charged",
                "quantity_model": "fractional units; no historical exchange lot/minimum reconstruction",
                "forced_final_liquidation": False, "fee_asset": "USDT", "bnb_fee_discount": False,
                "training": self.training, "episode_start": str(self.episode_start),
                "episode_end": str(self.episode_end), "completed_until": str(self.current_date),
                "gymnasium_installed": GYM_AVAILABLE,
                "training_window_quality": self.training_window_metadata,
                "training_episode_history": list(self.training_episode_history) if self.training else [],
            },
        }

    def render(self):
        return None

    def close(self):
        return None
