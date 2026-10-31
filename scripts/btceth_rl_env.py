"""Isolated BTC/ETH, cash-funded daily portfolio research environment.

Decisions see prior completed days only. Reductions fill at 00 UTC, increases
at 01 UTC, and equity/high-water marks use every completed hourly close.
This module neither reads credentials nor submits exchange orders.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from scripts.crypto_rl_portfolio_env import (
    CryptoPortfolioEnv, GYM_AVAILABLE, PortfolioLedger, _BaseEnv, spaces, utc,
    valid_price,
)


class BTCETHPortfolioEnv(_BaseEnv):
    """Two continuous controls: risk exposure, then ETH share of coin exposure.

    Targets are BTC, ETH, USDT. The full nonpositive exposure-action half-space
    means cash. Limits constrain requested targets; ordinary price drift can
    move actual weights between daily decisions. Training windows are selected
    on market/feature completeness alone and never on their realized returns.
    """

    metadata = {"render_modes": []}
    SYMBOLS = ("BTCUSDT", "ETHUSDT")
    COIN_FEATURES = (
        "close23utc_return_1d/.1", "close23utc_return_7d/.3", "close23utc_return_30d/.5",
        "close23utc_return_90d/1", "close23utc_return_180d/2", "close23utc_sma20_deviation/.2",
        "close23utc_sma60_deviation/.35", "close23utc_sma200_deviation/.5",
        "close23utc_volatility30/.1", "volume24h_relative30_minus1/2",
    )
    FEATURE_NAMES = tuple(f"{symbol}:{feature}" for symbol, feature in product(SYMBOLS, COIN_FEATURES)) + (
        "ETHBTC_return7/.3", "ETHBTC_return30/.5", "ETHBTC_return90/1",
        "BTC_weight", "ETH_weight", "cash_weight", "current_drawdown",
    )
    FILL_COLUMNS = CryptoPortfolioEnv.FILL_COLUMNS
    FEATURE_INPUT_POLICY = {
        "daily_close": "actual close of the 23UTC hour; missing 23UTC remains NaN; never resample-last fallback",
        "daily_volume": "sum of all 24 finite nonnegative hourly volume observations; incomplete days remain NaN",
        "feature_validity": "each feature requires only its actual causal close/volume inputs; no blanket 200-day full-hour condition",
        "execution_validity": "every trading day requires all 24 OHLCV hours for both coins; training episodes cannot cross gaps",
        "fill_policy": "no interpolation, forward fill, backward fill or future inputs",
    }

    def __init__(self, dataset, start, end, costs=None, capital=10000.,
                 training=False, episode_days=128, cash_floor=.05,
                 max_asset_weight=.95, rebalance_band=.025,
                 drawdown_penalty=.25, record_history=True):
        super().__init__()
        self.start, self.end = utc(start), utc(end)
        if self.start != self.start.normalize() or self.end != self.end.normalize() or self.start >= self.end:
            raise ValueError("Require nonempty UTC daily [start,end) boundaries")
        self.capital, self.training = float(capital), bool(training)
        self.episode_days, self.record_history = int(episode_days), bool(record_history)
        self.cash_floor, self.max_asset_weight = float(cash_floor), float(max_asset_weight)
        self.rebalance_band, self.drawdown_penalty = float(rebalance_band), float(drawdown_penalty)
        if not np.isfinite(self.capital) or self.capital <= 0 or self.episode_days < 1:
            raise ValueError("Positive finite capital and episode_days required")
        if not (0 <= self.cash_floor < 1 and 0 < self.max_asset_weight <= 1):
            raise ValueError("Invalid cash floor or asset cap")
        if not (np.isfinite(self.rebalance_band) and 0 <= self.rebalance_band <= 1
                and np.isfinite(self.drawdown_penalty) and self.drawdown_penalty >= 0):
            raise ValueError("Invalid rebalance band or drawdown penalty")
        self.costs = dict(costs or {"fee": .001, "slippage": .0002})
        PortfolioLedger(self.capital, **self.costs)  # Validate the declared costs.
        supplied_assets = dataset.get("assets", {})
        self.assets = {symbol: {**supplied_assets.get(symbol, {}), "symbol": symbol[:-4],
                                "binance_symbol": symbol} for symbol in self.SYMBOLS}
        self._frames, self._arrays = {}, {}
        reference = None
        for field in ("open", "high", "low", "close", "volume"):
            frame = dataset["hourly_" + field]
            if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
                raise ValueError("Market hours require timezone-aware DataFrames")
            if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
                raise ValueError("Market hour index must be unique and ordered")
            if frame.columns.has_duplicates or set(frame.columns) != set(self.SYMBOLS):
                raise ValueError("Market columns must be exactly BTCUSDT and ETHUSDT")
            index = frame.index.tz_convert("UTC")
            if (index != index.floor("h")).any():
                raise ValueError("Market timestamps must be hour opens")
            if reference is not None and not index.equals(reference):
                raise ValueError("All two-coin hourly frames must be time aligned")
            reference = index
            copied = frame.loc[:, list(self.SYMBOLS)].astype(float).copy()
            copied.index = index
            self._frames[field] = copied
            self._arrays[field] = copied.to_numpy()
        if not len(reference) or not reference.equals(pd.date_range(reference[0], reference[-1], freq="h")):
            raise ValueError("Hourly market grid contains omitted rows")
        if self.start < reference[0] or self.end > reference[-1] + pd.Timedelta(hours=1):
            raise ValueError("Episode lies outside market data")
        self._index, self._hour_lookup = reference, {t: i for i, t in enumerate(reference)}
        self._features, self._feature_valid, self._market_valid = self._make_features()
        self._eligible_starts, self.training_window_metadata = self._window_candidates()
        self.observation_space = spaces.Box(-10., 10., shape=(len(self.FEATURE_NAMES),), dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.training_episode_history = []
        self._done, self.ledger = True, None

    def _make_features(self):
        complete = np.ones_like(self._arrays["close"], dtype=bool)
        for field, values in self._arrays.items():
            complete &= np.isfinite(values) & (values >= 0 if field == "volume" else values > 0)
        count = pd.DataFrame(complete, index=self._index, columns=self.SYMBOLS).resample("D").sum()
        market_valid = count.eq(24).all(axis=1)
        # Intraday exchange outages do not erase a subsequently observed real
        # 23UTC close. Conversely resample.last() must never substitute 22UTC
        # when the actual closing hour is absent.
        closes = self._frames["close"].loc[self._index.hour == 23].copy()
        closes.index = closes.index.normalize()
        closes = closes.reindex(count.index)
        closes = closes.where(np.isfinite(closes) & closes.gt(0))
        hourly_volume = self._frames["volume"]
        good_volume = np.isfinite(hourly_volume) & hourly_volume.ge(0)
        volume_count = good_volume.resample("D").sum()
        volume = hourly_volume.where(good_volume).resample("D").sum(min_count=24).where(volume_count.eq(24))
        ret = closes.pct_change(fill_method=None)
        pieces = [
            ret / .1, (closes / closes.shift(7) - 1) / .3,
            (closes / closes.shift(30) - 1) / .5,
            (closes / closes.shift(90) - 1), (closes / closes.shift(180) - 1) / 2,
            (closes / closes.rolling(20, min_periods=20).mean() - 1) / .2,
            (closes / closes.rolling(60, min_periods=60).mean() - 1) / .35,
            (closes / closes.rolling(200, min_periods=200).mean() - 1) / .5,
            ret.rolling(30, min_periods=30).std(ddof=1) / .1,
            (volume / volume.rolling(30, min_periods=30).mean() - 1) / 2,
        ]
        coin = np.stack([piece.to_numpy() for piece in pieces], axis=2).reshape(len(closes), -1)
        ratio = closes.ETHUSDT / closes.BTCUSDT
        relative = np.column_stack([(ratio / ratio.shift(n) - 1).to_numpy() / scale
                                    for n, scale in ((7, .3), (30, .5), (90, 1.))])
        values = np.column_stack([coin, relative])
        valid = np.isfinite(values).all(axis=1)
        days = closes.index + pd.Timedelta(days=1)
        # Invalid histories remain invalid, rather than fabricated feature zeros.
        return ({day: np.clip(values[i], -10, 10) for i, day in enumerate(days)},
                dict(zip(days, valid)), market_valid.to_dict())

    def _window_candidates(self):
        days = pd.date_range(self.start, self.end, freq="D", inclusive="left")
        bad = {day: [name for name, good in (
            ("incomplete_feature_input_dependencies", self._feature_valid.get(day, False)),
            ("incomplete_current_day_hourly_market", self._market_valid.get(day, False))) if not good]
            for day in days}
        bad = {day: reasons for day, reasons in bad.items() if reasons}
        if not self.training:
            if bad:
                day = next(iter(bad))
                raise ValueError(f"Evaluation must be complete; no day removal or interpolation: {day}: {bad[day]}")
            return [], {"applied": False, "reason": "evaluation continuous and unfiltered"}
        length, episode = len(days), min(len(days), self.episode_days)
        candidates = [days[i] for i in range(length - episode + 1)
                      if not any(day in bad for day in days[i:i + episode])]
        if not candidates:
            raise ValueError("No complete training episodes; never interpolate missing data")
        covered = set(day for first in candidates for day in pd.date_range(first, periods=episode, freq="D"))
        return candidates, {
            "applied": True, "selection_basis": "TRAIN market and causal feature completeness only; never PNL",
            "requested_train_days": length, "episode_days": episode,
            "candidate_windows": length - episode + 1, "eligible_windows": len(candidates),
            "excluded_windows": length - episode + 1 - len(candidates),
            "covered_training_days": len(covered), "earliest_eligible_window": str(candidates[0]),
            "latest_eligible_window": str(candidates[-1]),
            "complete_feature_days": sum(bool(self._feature_valid.get(day, False)) for day in days),
            "complete_market_days": sum(bool(self._market_valid.get(day, False)) for day in days),
            "incomplete_days": [{"date": str(day), "reasons": reasons} for day, reasons in bad.items()],
        }

    def _prices(self, timestamp, field="close"):
        at = self._hour_lookup.get(utc(timestamp))
        if at is None:
            raise ValueError(f"No market hour: {timestamp}")
        values = self._arrays[field][at]
        if not all(valid_price(value) for value in values):
            raise ValueError("Missing/nonpositive two-coin execution or mark price")
        return dict(zip(self.SYMBOLS, values))

    @property
    def current_weights(self):
        if self.ledger is None:
            raise RuntimeError("Reset before requesting portfolio weights")
        prices = self._prices(self.current_date - pd.Timedelta(hours=1))
        nav = self.ledger.value(prices)
        coins = [self.ledger.positions.get(symbol, 0.) * prices[symbol] / nav for symbol in self.SYMBOLS]
        return np.r_[coins, self.ledger.cash / nav]

    @property
    def current_drawdown(self):
        return max(0., 1 - self._nav / self._high_water)

    def _observation(self):
        if not self._feature_valid.get(self.current_date, False):
            raise ValueError(f"Insufficient causal features at {self.current_date}")
        observation = np.r_[self._features[self.current_date], self.current_weights, self.current_drawdown]
        if not np.isfinite(observation).all():
            raise ValueError("Non-finite observation")
        return np.clip(observation, -10, 10).astype(np.float32)

    def action_to_weights(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Action requires two finite values")
        clipped = np.clip(action, -1., 1.)
        gross = (1 - self.cash_floor) * max(float(clipped[0]), 0.)
        eth_share = (float(clipped[1]) + 1) / 2
        coins = np.minimum([gross * (1 - eth_share), gross * eth_share], self.max_asset_weight)
        return np.r_[coins, 1 - np.sum(coins)]

    def _validate_target(self, weights):
        value = np.asarray(weights, dtype=float).copy()
        if value.shape != (3,) or not np.isfinite(value).all() or (value < -1e-12).any():
            raise ValueError("Target must have three finite nonnegative weights")
        if abs(value.sum() - 1) > 1e-8 or value[2] < self.cash_floor - 1e-10:
            raise ValueError("Target must sum to one and respect cash floor")
        if (value[:2] > self.max_asset_weight + 1e-10).any():
            raise ValueError("Target exceeds asset cap")
        value = np.maximum(value, 0.)
        value[2] = 1 - value[:2].sum()
        return value

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.training:
            self.episode_start = self._eligible_starts[int(self.np_random.integers(len(self._eligible_starts)))]
            self.episode_end = self.episode_start + pd.Timedelta(days=min(self.episode_days, (self.end - self.start).days))
        else:
            self.episode_start, self.episode_end = self.start, self.end
        self.current_date, self._nav, self._high_water = self.episode_start, self.capital, self.capital
        self.ledger = PortfolioLedger(self.capital, **self.costs, assets=self.assets)
        self._equity_rows, self._weight_rows = [], []
        self._boundary_rows = [{"timestamp": self.current_date, "equity": self.capital, "reward": 0.}]
        self._fees = self._notional = self._slippage_cost = 0.
        self._fill_count = self._steps = 0
        self._last_prices, self._done = self._prices(self.current_date - pd.Timedelta(hours=1)), False
        if self.training:
            self.training_episode_history.append({"start": str(self.episode_start), "end_exclusive": str(self.episode_end),
                "completed_until": str(self.episode_start), "completed_steps": 0, "terminated": False, "truncated": False})
        return self._observation(), {"date": self.current_date, "equity": self._nav}

    def _mark(self, timestamp):
        prices = self._prices(timestamp)
        nav = self.ledger.value(prices)
        self._high_water = max(self._high_water, nav)
        self._day_deepest_drawdown = max(self._day_deepest_drawdown, 1 - nav / self._high_water)
        if self.record_history:
            self._equity_rows.append({"timestamp": timestamp, "equity": nav, "cash": self.ledger.cash,
                "base_value": nav - self.ledger.cash, "exposure": (nav - self.ledger.cash) / nav})
        self._last_prices, self._last_mark_nav = prices, nav

    def _rebalance(self, weights):
        targets = dict(zip(self.SYMBOLS, weights[:2]))
        p0 = self._prices(self.current_date, "open")
        initial = self.ledger.value(p0)
        current = {key: quantity * p0[key] for key, quantity in self.ledger.positions.items()}
        sell_volume = lambda nav: sum(max(value - targets[key] * nav, 0.) for key, value in current.items())
        sell_cost = 1 - (1 - self.ledger.slippage) * (1 - self.ledger.fee)
        after_sell = CryptoPortfolioEnv._solve_nav(initial, sell_cost, sell_volume)
        for key, value in current.items():
            reduction = max(value - targets[key] * after_sell, 0.)
            if reduction > 1e-9:
                self.ledger.sell(key, min(reduction / p0[key], self.ledger.positions[key]), p0[key], self.current_date)
        self._mark(self.current_date)
        buy_time = self.current_date + pd.Timedelta(hours=1)
        p1 = self._prices(buy_time, "open")
        initial = self.ledger.value(p1)
        held = {key: quantity * p1[key] for key, quantity in self.ledger.positions.items()}
        buy_volume = lambda nav: sum(max(weight * nav - held.get(key, 0.), 0.) for key, weight in targets.items())
        buy_cost = (1 + self.ledger.slippage) * (1 + self.ledger.fee) - 1
        after_buy = CryptoPortfolioEnv._solve_nav(initial, buy_cost, buy_volume)
        desired = {key: max(weight * after_buy - held.get(key, 0.), 0.) for key, weight in targets.items()}
        total = sum(desired.values())
        reserve = max(self.cash_floor, weights[2])
        affordable = max(0., (self.ledger.cash - reserve * initial) / (1 + buy_cost - reserve * buy_cost))
        scale = min(1., affordable / total) if total > 0 else 0.
        for key, value in desired.items():
            if value * scale > 1e-9:
                self.ledger.buy(key, value * scale / p1[key], p1[key], buy_time)

    def step(self, action):
        return self.step_target(self.action_to_weights(action))

    def step_target(self, weights):
        if self._done:
            raise RuntimeError("Episode ended; reset before another step")
        target = None if weights is None else self._validate_target(weights)
        before, initial_drawdown = self._nav, self.current_drawdown
        self._day_deepest_drawdown = initial_drawdown
        actual = self.current_weights
        force_exit = target is not None and target[:2].sum() <= 1e-12 and actual[:2].sum() > 0
        execute = target is not None and (force_exit or np.max(np.abs(target - actual)) >= self.rebalance_band)
        fill_start = len(self.ledger.fills)
        if execute:
            self._rebalance(target)
        for hour in range(1 if execute else 0, 24):
            self._mark(self.current_date + pd.Timedelta(hours=hour))
        new_fills = self.ledger.fills[fill_start:]
        self._fees += sum(fill["fee"] for fill in new_fills)
        self._notional += sum(fill["notional"] for fill in new_fills)
        self._slippage_cost += sum(fill["slippage_cost"] for fill in new_fills)
        self._fill_count += len(new_fills)
        final_nav = self._last_mark_nav
        if final_nav <= 0:
            raise ValueError("Nonpositive spot portfolio equity")
        increase = max(self._day_deepest_drawdown - initial_drawdown, 0.)
        log_return = float(np.log(final_nav / before))
        reward = float(100 * (log_return - self.drawdown_penalty * increase))
        if self.record_history:
            for i, key in enumerate(self.SYMBOLS):
                weight = self.ledger.positions.get(key, 0.) * self._last_prices[key] / final_nav
                self._weight_rows.append({"date": self.current_date, "slot": i, "asset_key": key,
                    "symbol": key[:-4], "target_weight": None if target is None else target[i],
                    "actual_weight": weight, "rebalanced": bool(execute)})
        else:
            self.ledger.fills.clear()
        self.current_date += pd.Timedelta(days=1)
        self._nav, self._steps = final_nav, self._steps + 1
        self._done = self.current_date >= self.episode_end
        # Random training fragments end because of a time limit, including the
        # fragment that happens to reach the right edge of the training data.
        # Market interaction itself has no absorbing terminal state there.
        truncated = self._done and self.training
        terminated = self._done and not self.training
        if self.record_history:
            self._boundary_rows.append({"timestamp": self.current_date, "equity": final_nav, "reward": reward})
        if self.training:
            self.training_episode_history[-1].update(completed_until=str(self.current_date),
                completed_steps=self._steps, terminated=bool(terminated), truncated=bool(truncated))
        info = {"date": self.current_date, "equity": final_nav, "cash": self.ledger.cash,
                "target_weights": target, "rebalanced": bool(execute), "log_net_return": log_return,
                "drawdown_increase": increase, "current_drawdown": self.current_drawdown,
                "hourly_high_water": self._high_water}
        return self._observation(), reward, bool(terminated), bool(truncated), info

    def step_hold(self):
        return self.step_target(None)

    def get_result(self):
        if not self._steps:
            raise RuntimeError("No completed daily steps")
        equity = pd.DataFrame(self._equity_rows, columns=("timestamp", "equity", "cash", "base_value", "exposure"))
        equity["timestamp"] = pd.to_datetime(equity.timestamp, utc=True)
        equity = equity.set_index("timestamp")
        boundary = pd.DataFrame(self._boundary_rows).set_index("timestamp")
        final, cash = self.ledger.value(self._last_prices), self.ledger.cash
        unrealized = sum(quantity * self._last_prices[key] - self.ledger.cost_basis[key]
                         for key, quantity in self.ledger.positions.items())
        if abs(final - self.capital - self.ledger.realized_pnl - unrealized) > max(1e-6, self.capital * 1e-10):
            raise AssertionError("Realized/open PNL bridge failed")
        holdings = [{"asset_key": key, "symbol": key[:-4], "qty": quantity,
                     "mark_price": self._last_prices[key], "market_value": quantity * self._last_prices[key],
                     "weight": quantity * self._last_prices[key] / final}
                    for key, quantity in self.ledger.positions.items()]
        return {"equity": equity, "fills": pd.DataFrame(self.ledger.fills, columns=self.FILL_COLUMNS),
            "weights": pd.DataFrame(self._weight_rows), "boundary_equity": boundary,
            "metadata": {"initial_capital": self.capital, "final_equity": final, "final_cash": cash,
                "final_positions": dict(self.ledger.positions), "final_holdings": holdings,
                "fees": self._fees, "total_notional": self._notional, "slippage_cost": self._slippage_cost,
                "fill_count": self._fill_count, "realized_pnl": self.ledger.realized_pnl, "unrealized_pnl": unrealized,
                "final_liquidation_equity": cash + (final - cash) * (1 - self.ledger.slippage) * (1 - self.ledger.fee),
                "fee_rate": self.ledger.fee, "slippage": self.ledger.slippage, "fee_asset": "USDT",
                "bnb_fee_discount": False, "cash_floor_target": self.cash_floor,
                "max_asset_weight_target": self.max_asset_weight, "rebalance_band": self.rebalance_band,
                "drawdown_penalty": self.drawdown_penalty,
                "reward_formula": "100*(log(net_close_equity_t/net_close_equity_t-1)-penalty*max(day_hourly_deepest_drawdown-start_drawdown,0))",
                "valuation_frequency": "1h", "equity_timestamp_semantics": "hour_open label, marked at hour_close",
                "execution_timing": "daily 00UTC reductions, 01UTC purchases; each leg charged",
                "quantity_model": "fractional units; no historical exchange lot/minimum reconstruction",
                "forced_final_liquidation": False, "record_history": self.record_history,
                "training": self.training, "episode_start": str(self.episode_start), "episode_end": str(self.episode_end),
                "completed_until": str(self.current_date), "gymnasium_installed": GYM_AVAILABLE,
                "training_window_quality": self.training_window_metadata,
                "feature_input_policy": dict(self.FEATURE_INPUT_POLICY),
                "training_episode_history": list(self.training_episode_history) if self.training else [],
                "hourly_high_water": self._high_water, "current_drawdown": self.current_drawdown}}

    def render(self):
        return None

    def close(self):
        return None
