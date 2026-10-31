"""Independent, single-pair spot RL environment for BTCUSDT or ETHUSDT.

One policy controls one coin and its own USDT cash account. The other coin is
neither a feature nor an inventory slot. Financial primitives and hourly
valuation follow the frozen BTC/ETH research environment; no orders are sent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.btceth_rl_env import BTCETHPortfolioEnv
from scripts.crypto_rl_portfolio_env import (
    CryptoPortfolioEnv, PortfolioLedger, _BaseEnv, spaces, utc,
)


class SinglePairEnv(BTCETHPortfolioEnv):
    """13 observations, one continuous action, and exactly one allowed coin.

    Action ``a`` maps to ``min(max_asset_weight, (1-cash_floor)*max(a,0))``
    after clipping to [-1,1]. With the study defaults this is .95*max(a,0).
    A nonpositive action requests cash; no borrowing or shorting is possible.
    """

    SUPPORTED_SYMBOLS = ("BTCUSDT", "ETHUSDT")
    SYMBOLS = ()  # Bound to exactly (symbol,) on each constructed account.
    FEATURE_NAMES = tuple(f"coin:{name}" for name in BTCETHPortfolioEnv.COIN_FEATURES) + (
        "coin_weight", "cash_weight", "current_drawdown",
    )
    FEATURE_INPUT_POLICY = {
        "daily_close": "selected coin's actual 23UTC hour close; missing 23UTC remains NaN; no resample-last fallback",
        "daily_volume": "selected coin's 24 finite nonnegative hourly volume observations; incomplete days remain NaN",
        "feature_validity": "only the selected coin's causal close/volume dependencies; no cross-coin or ratio feature",
        "execution_validity": "every trading day requires all 24 OHLCV hours of the selected coin only; episodes cannot cross gaps",
        "fill_policy": "no interpolation, forward fill, backward fill or future inputs",
    }

    def __init__(self, dataset, symbol, start, end, costs=None, capital=10000.,
                 training=False, episode_days=128, cash_floor=.05,
                 max_asset_weight=.95, rebalance_band=.025,
                 drawdown_penalty=.25, record_history=True):
        # The frozen parent's initializer assumes two assets. Initialize only
        # its Gym base, then construct a genuinely one-column market state.
        _BaseEnv.__init__(self)
        if not isinstance(symbol, str) or symbol not in self.SUPPORTED_SYMBOLS:
            raise ValueError("Single-pair research supports BTCUSDT or ETHUSDT only")
        self.symbol = symbol
        self.SYMBOLS = (symbol,)
        self.FEATURE_NAMES = tuple(f"{symbol}:{name}" for name in self.COIN_FEATURES) + (
            "coin_weight", "cash_weight", "current_drawdown",
        )
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
        PortfolioLedger(self.capital, **self.costs)
        supplied = dataset.get("assets", {}).get(symbol, {})
        self.assets = {symbol: {**supplied, "symbol": symbol[:-4], "binance_symbol": symbol}}
        self._frames, self._arrays = {}, {}
        reference = None
        for field in ("open", "high", "low", "close", "volume"):
            frame = dataset["hourly_" + field]
            if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
                raise ValueError("Market hours require timezone-aware DataFrames")
            if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
                raise ValueError("Market hour index must be unique and ordered")
            if list(frame.columns).count(symbol) != 1:
                raise ValueError("Selected market column must exist exactly once")
            # Ignore every other column, including its dtype, NaNs and values.
            # All feature and availability calculations use this private copy.
            copied = frame.loc[:, [symbol]].astype(float).copy()
            index = copied.index.tz_convert("UTC")
            if (index != index.floor("h")).any():
                raise ValueError("Market timestamps must be hour opens")
            if reference is not None and not index.equals(reference):
                raise ValueError("Selected coin OHLCV frames must be time aligned")
            reference = index
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
        self.observation_space = spaces.Box(-10., 10., shape=(13,), dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., shape=(1,), dtype=np.float32)
        self.training_episode_history = []
        self._done, self.ledger = True, None

    def _make_features(self):
        complete = np.ones_like(self._arrays["close"], dtype=bool)
        for field, values in self._arrays.items():
            complete &= np.isfinite(values) & (values >= 0 if field == "volume" else values > 0)
        count = pd.DataFrame(complete, index=self._index, columns=self.SYMBOLS).resample("D").sum()
        market_valid = count.eq(24).all(axis=1)
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
        values = np.column_stack([piece.to_numpy()[:, 0] for piece in pieces])
        valid = np.isfinite(values).all(axis=1)
        days = closes.index + pd.Timedelta(days=1)
        return ({day: np.clip(values[i], -10, 10) for i, day in enumerate(days)},
                dict(zip(days, valid)), market_valid.to_dict())

    def _assert_own_inventory(self):
        if self.ledger is None:
            return
        allowed = {self.symbol}
        if not set(self.ledger.positions).issubset(allowed) or not set(self.ledger.cost_basis).issubset(allowed):
            raise ValueError("Single-pair account contains foreign inventory; no automatic sale or offset allowed")

    @property
    def current_weights(self):
        self._assert_own_inventory()
        return super().current_weights

    def _observation(self):
        self._assert_own_inventory()
        return super()._observation()

    def _mark(self, timestamp):
        self._assert_own_inventory()
        return super()._mark(timestamp)

    def action_to_weights(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("Single-pair action requires one finite value")
        clipped = float(np.clip(action[0], -1., 1.))
        coin = min(self.max_asset_weight, (1 - self.cash_floor) * max(clipped, 0.))
        return np.array([coin, 1 - coin], dtype=float)

    def _validate_target(self, weights):
        value = np.asarray(weights, dtype=float).copy()
        if value.shape != (2,) or not np.isfinite(value).all() or (value < -1e-12).any():
            raise ValueError("Single-pair target requires [coin,cash], both finite and nonnegative")
        if abs(value.sum() - 1) > 1e-8 or value[1] < self.cash_floor - 1e-10:
            raise ValueError("Target must sum to one and respect cash floor")
        if value[0] > self.max_asset_weight + 1e-10:
            raise ValueError("Target exceeds asset cap")
        value = np.maximum(value, 0.)
        value[1] = 1 - value[0]
        return value

    def _rebalance(self, weights):
        self._assert_own_inventory()
        targets = {self.symbol: weights[0]}
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
        reserve = max(self.cash_floor, weights[1])
        affordable = max(0., (self.ledger.cash - reserve * initial) / (1 + buy_cost - reserve * buy_cost))
        scale = min(1., affordable / total) if total > 0 else 0.
        for key, value in desired.items():
            if value * scale > 1e-9:
                self.ledger.buy(key, value * scale / p1[key], p1[key], buy_time)

    def step_target(self, weights):
        self._assert_own_inventory()
        if self._done:
            raise RuntimeError("Episode ended; reset before another step")
        target = None if weights is None else self._validate_target(weights)
        before, initial_drawdown = self._nav, self.current_drawdown
        self._day_deepest_drawdown = initial_drawdown
        actual = self.current_weights
        force_exit = target is not None and target[0] <= 1e-12 and actual[0] > 0
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
            raise ValueError("Nonpositive single-pair spot equity")
        increase = max(self._day_deepest_drawdown - initial_drawdown, 0.)
        log_return = float(np.log(final_nav / before))
        reward = float(100 * (log_return - self.drawdown_penalty * increase))
        if self.record_history:
            weight = self.ledger.positions.get(self.symbol, 0.) * self._last_prices[self.symbol] / final_nav
            self._weight_rows.append({"date": self.current_date, "slot": 0, "asset_key": self.symbol,
                "symbol": self.symbol[:-4], "target_weight": None if target is None else target[0],
                "actual_weight": weight, "rebalanced": bool(execute)})
        else:
            self.ledger.fills.clear()
        self.current_date += pd.Timedelta(days=1)
        self._nav, self._steps = final_nav, self._steps + 1
        self._done = self.current_date >= self.episode_end
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
                "hourly_high_water": self._high_water, "pair": self.symbol}
        return self._observation(), reward, bool(terminated), bool(truncated), info

    def get_result(self):
        self._assert_own_inventory()
        result = super().get_result()
        result["metadata"].update(pair=self.symbol, independent_capital=self.capital,
            no_ensemble=True, tradable_symbols=list(self.SYMBOLS), cash_symbol="USDT",
            observation_features=list(self.FEATURE_NAMES), observation_dim=13, action_dim=1,
            action_mapping="coin=min(asset_cap,(1-cash_floor)*max(clip(action[0],-1,1),0)); residual own USDT",
            initial_state="independent all-cash account; no inventory or cash shared with other pairs")
        return result
