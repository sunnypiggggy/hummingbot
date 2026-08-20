from decimal import Decimal
from typing import List

from pydantic import Field, field_validator

from controllers.market_making.dman_maker_v2 import (
    DManMakerV2 as _DManMakerV2,
    DManMakerV2Config as _DManMakerV2Config,
)
from hummingbot.core.data_type.common import TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.models.executor_actions import (
    CreateExecutorAction,
    ExecutorAction,
    StopExecutorAction,
)
from hummingbot.strategy_v2.models.executors import CloseType


class DManMakerV3MacroConfig(_DManMakerV2Config):
    controller_name: str = "dman_maker_v3_macro"
    macro_buy_enabled: bool = Field(
        default=True, json_schema_extra={"is_updatable": True}
    )
    macro_sell_enabled: bool = Field(
        default=True, json_schema_extra={"is_updatable": True}
    )
    long_only_enabled: bool = Field(
        default=False, json_schema_extra={"is_updatable": True}
    )
    macro_decision_id: str = Field(
        default="", json_schema_extra={"is_updatable": True}
    )
    policy_version: str = Field(
        default="dca-macro-v3", json_schema_extra={"is_updatable": True}
    )
    total_amount_quote: Decimal = Field(
        default=Decimal("190"),
        gt=0,
        le=190,
        json_schema_extra={"is_updatable": True},
    )
    dca_spreads: List[Decimal] = Field(
        default="0.01,0.02,0.04,0.08",
        json_schema_extra={"is_updatable": True},
    )
    dca_amounts: List[Decimal] = Field(
        default="0.1,0.2,0.3,0.4",
        json_schema_extra={"is_updatable": True},
    )
    take_profit: Decimal = Field(
        default=Decimal("0.02"), json_schema_extra={"is_updatable": True}
    )
    executor_refresh_time: float = Field(
        default=18000, json_schema_extra={"is_updatable": True}
    )
    time_limit_from_first_fill: bool = Field(
        default=True, json_schema_extra={"is_updatable": True}
    )
    stop_loss_on_partial_fills: bool = Field(
        default=True, json_schema_extra={"is_updatable": True}
    )
    shutdown_retry_seconds: float = Field(
        default=1.0, ge=0.2, le=5.0, json_schema_extra={"is_updatable": True}
    )
    sell_trend_gate_enabled: bool = Field(
        default=True, json_schema_extra={"is_updatable": True}
    )
    sell_trend_interval: str = Field(default="5m")
    sell_trend_fast_ema: int = Field(default=12, ge=3, le=100)
    sell_trend_slow_ema: int = Field(default=48, ge=12, le=300)
    sell_trend_roc_bars: int = Field(default=12, ge=3, le=100)
    sell_trend_trigger_roc: Decimal = Field(
        default=Decimal("0.006"), ge=0, le=Decimal("0.05"),
        json_schema_extra={"is_updatable": True},
    )
    sell_trend_trigger_ema_gap: Decimal = Field(
        default=Decimal("0.002"), ge=0, le=Decimal("0.03"),
        json_schema_extra={"is_updatable": True},
    )
    sell_trend_recovery_roc: Decimal = Field(
        default=Decimal("0.002"), ge=-Decimal("0.05"), le=Decimal("0.02"),
        json_schema_extra={"is_updatable": True},
    )
    sell_trend_recovery_bars: int = Field(
        default=3, ge=1, le=24, json_schema_extra={"is_updatable": True}
    )
    sell_stop_cooldown_seconds: int = Field(
        default=1800, ge=1800, le=21600, json_schema_extra={"is_updatable": True}
    )
    sell_stop_event_at: float = Field(
        default=0.0, ge=0.0, json_schema_extra={"is_updatable": True}
    )
    sell_stop_event_id: str = Field(
        default="", json_schema_extra={"is_updatable": True}
    )

    @field_validator("dca_spreads")
    @classmethod
    def validate_macro_spreads(cls, value):
        normalized = [Decimal(str(item)) for item in value]
        if len(normalized) != 4 or normalized != sorted(normalized):
            raise ValueError("macro DCA requires four ascending spreads")
        return value

    @field_validator("dca_amounts", mode="after")
    @classmethod
    def validate_macro_amounts(cls, value):
        normalized = [Decimal(str(item)) for item in value]
        if len(normalized) != 4 or any(item <= 0 for item in normalized):
            raise ValueError("macro DCA requires four positive weights")
        return value

    @field_validator("sell_trend_slow_ema")
    @classmethod
    def validate_sell_trend_ema_order(cls, value, info):
        fast = int(info.data.get("sell_trend_fast_ema", 12))
        if int(value) <= fast:
            raise ValueError("sell trend slow EMA must exceed fast EMA")
        return value


class DManMakerV3Macro(_DManMakerV2):
    """DMan Maker with independently controlled BUY and SELL risk gates."""

    def __init__(self, config: DManMakerV3MacroConfig, *args, **kwargs):
        self.config = config
        self.max_records = max(
            config.sell_trend_slow_ema + 20,
            config.sell_trend_roc_bars + config.sell_trend_recovery_bars + 5,
        )
        self._sell_trend_blocked = False
        self._sell_trend_reason = "not_triggered"
        self._sell_trend_recovery_count = 0
        self._sell_stop_cooldown_until = 0.0
        self._sell_trend_last_candle = None
        self._sell_trend_metrics = {}
        self._sell_trend_data_healthy = False
        self._observed_stop_loss_ids = {}
        self._sell_stop_event_at = 0.0
        self._sell_stop_event_id = ""
        super().__init__(config, *args, **kwargs)
        self.config = config
        self._apply_durable_sell_stop_event(
            float(config.sell_stop_event_at), str(config.sell_stop_event_id),
        )

    @property
    def macro_buy_enabled(self) -> bool:
        return self.config.macro_buy_enabled

    @property
    def sell_creation_enabled(self) -> bool:
        return not self.config.long_only_enabled and self.config.macro_sell_enabled and not (
            self.config.sell_trend_gate_enabled and self._sell_trend_blocked
        )

    def creation_side_enabled(self, side: TradeType) -> bool:
        if side == TradeType.BUY:
            return self.macro_buy_enabled
        if side == TradeType.SELL:
            return self.sell_creation_enabled
        return False

    def force_stop_side_required(self, side: TradeType) -> bool:
        """Risk gates may stop open exposure; long-only migration may not."""
        if side == TradeType.BUY:
            return not self.macro_buy_enabled
        if side == TradeType.SELL:
            return not self.config.macro_sell_enabled
        return True

    def get_candles_config(self) -> List[CandlesConfig]:
        if self.config.long_only_enabled or not self.config.sell_trend_gate_enabled:
            return []
        return [CandlesConfig(
            connector=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            interval=self.config.sell_trend_interval,
            max_records=self.max_records,
        )]

    def _observe_stop_losses(self, now: float) -> None:
        for executor in self.executors_info:
            if executor.close_type != CloseType.STOP_LOSS:
                continue
            event_at = float(executor.close_timestamp or now)
            if executor.id in self._observed_stop_loss_ids:
                continue
            self._observed_stop_loss_ids[executor.id] = event_at
            if executor.side == TradeType.SELL:
                self._apply_durable_sell_stop_event(event_at, executor.id)
        cutoff = now - 7 * 86400
        self._observed_stop_loss_ids = {
            executor_id: event_at
            for executor_id, event_at in self._observed_stop_loss_ids.items()
            if event_at >= cutoff
        }

    def _apply_durable_sell_stop_event(self, event_at: float, event_id: str) -> None:
        event_at = float(event_at or 0)
        event_id = str(event_id or "")
        if event_at <= 0:
            return
        is_new = event_at > self._sell_stop_event_at or (
            event_at == self._sell_stop_event_at
            and event_id
            and event_id != self._sell_stop_event_id
        )
        if not is_new:
            return
        self._sell_stop_event_at = event_at
        self._sell_stop_event_id = event_id
        self._sell_trend_blocked = True
        self._sell_trend_reason = "sell_stop_loss_recovery"
        self._sell_trend_recovery_count = 0
        self._sell_stop_cooldown_until = max(
            self._sell_stop_cooldown_until,
            event_at + self.config.sell_stop_cooldown_seconds,
        )

    async def update_processed_data(self):
        await super().update_processed_data()
        now = float(self.market_data_provider.time())
        self._observe_stop_losses(now)
        if self.config.long_only_enabled:
            self._sell_trend_data_healthy = True
            self._sell_trend_reason = "long_only_policy"
            self._sell_trend_metrics = {"strategy_mode": "LONG_ONLY"}
            return
        if not self.config.sell_trend_gate_enabled:
            self._sell_trend_data_healthy = True
            return
        try:
            candles = self.market_data_provider.get_candles_df(
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                interval=self.config.sell_trend_interval,
                max_records=self.max_records,
            )
            # The final feed row may still be forming. Every decision therefore
            # uses only candles whose close is already immutable.
            completed = candles.iloc[:-1].copy()
            required = max(self.config.sell_trend_slow_ema, self.config.sell_trend_roc_bars + 1)
            if len(completed) < required:
                raise ValueError(f"only {len(completed)} completed candles; need {required}")
            close = completed["close"].astype(float)
            fast = close.ewm(span=self.config.sell_trend_fast_ema, adjust=False).mean()
            slow = close.ewm(span=self.config.sell_trend_slow_ema, adjust=False).mean()
            roc = close.iloc[-1] / close.iloc[-1 - self.config.sell_trend_roc_bars] - 1
            ema_gap = fast.iloc[-1] / slow.iloc[-1] - 1
            trigger = (
                close.iloc[-1] > slow.iloc[-1]
                and ema_gap >= float(self.config.sell_trend_trigger_ema_gap)
                and roc >= float(self.config.sell_trend_trigger_roc)
            )
            recovery = (
                fast.iloc[-1] <= slow.iloc[-1]
                or roc <= float(self.config.sell_trend_recovery_roc)
            )
            candle_id = str(completed.index[-1])
            if "timestamp" in completed.columns:
                candle_id = str(completed["timestamp"].iloc[-1])
            if candle_id != self._sell_trend_last_candle:
                self._sell_trend_last_candle = candle_id
                if trigger:
                    self._sell_trend_blocked = True
                    self._sell_trend_reason = "strong_uptrend"
                    self._sell_trend_recovery_count = 0
                elif self._sell_trend_blocked:
                    self._sell_trend_recovery_count = (
                        self._sell_trend_recovery_count + 1 if recovery else 0
                    )
                    if (
                        self._sell_trend_recovery_count >= self.config.sell_trend_recovery_bars
                        and now >= self._sell_stop_cooldown_until
                    ):
                        self._sell_trend_blocked = False
                        self._sell_trend_reason = "trend_recovered"
                        self._sell_trend_recovery_count = 0
            self._sell_trend_metrics = {
                "close": float(close.iloc[-1]),
                "fast_ema": float(fast.iloc[-1]),
                "slow_ema": float(slow.iloc[-1]),
                "roc": float(roc),
                "ema_gap": float(ema_gap),
            }
            self._sell_trend_data_healthy = True
        except Exception as exc:
            # A technical profit filter must never turn a transient candle read
            # failure into a portfolio liquidation. Retain an already active
            # SELL recovery hold, otherwise fall back to the original strategy.
            self._sell_trend_data_healthy = False
            self._sell_trend_metrics = {"error": str(exc)}

    def update_config(self, new_config: DManMakerV3MacroConfig):
        if new_config.total_amount_quote > Decimal("190"):
            raise ValueError("macro update cannot increase the 190 USDT cap")
        if new_config.stop_loss != Decimal("0.05"):
            raise ValueError("macro update cannot change the 5% stop loss")
        if new_config.time_limit != 18000:
            raise ValueError("macro update cannot change the 18000 second time limit")
        if new_config.executor_refresh_time != 18000:
            raise ValueError("macro update cannot change the 18000 second executor refresh")
        if not new_config.time_limit_from_first_fill:
            raise ValueError("macro DCA time limit must start at first fill")
        if not new_config.stop_loss_on_partial_fills:
            raise ValueError("macro DCA must protect partial fills with stop loss")
        if new_config.shutdown_retry_seconds != 1.0:
            raise ValueError("macro DCA shutdown verification must run every second")
        if not new_config.sell_trend_gate_enabled:
            raise ValueError("macro DCA SELL trend gate must remain enabled")
        if not new_config.long_only_enabled:
            raise ValueError("live DCA must remain in long-only mode")
        if new_config.sell_stop_cooldown_seconds not in {1800, 7200, 21600}:
            raise ValueError("SELL stop cooldown must be a validated 30m, 2h, or 6h value")
        if new_config.policy_version != "dca-macro-v3":
            raise ValueError("unsupported macro policy version")
        self._apply_durable_sell_stop_event(
            float(new_config.sell_stop_event_at), str(new_config.sell_stop_event_id),
        )
        super().update_config(new_config)
        self.dca_amounts_pct = [
            Decimal(amount) / sum(self.config.dca_amounts)
            for amount in self.config.dca_amounts
        ]
        self.spreads = self.config.dca_spreads

    def get_executor_config(self, level_id: str, price: Decimal, amount: Decimal):
        config = super().get_executor_config(level_id, price, amount)
        config.time_limit_from_first_fill = self.config.time_limit_from_first_fill
        config.stop_loss_on_partial_fills = self.config.stop_loss_on_partial_fills
        config.shutdown_retry_seconds = self.config.shutdown_retry_seconds
        return config

    def create_actions_proposal(self) -> List[ExecutorAction]:
        actions = super().create_actions_proposal()
        return [
            action
            for action in actions
            if not (
                isinstance(action, CreateExecutorAction)
                and not self.creation_side_enabled(
                    getattr(action.executor_config, "side", None)
                )
            )
        ]

    def executors_to_early_stop(self) -> List[ExecutorAction]:
        return [
            StopExecutorAction(
                controller_id=self.config.id,
                executor_id=executor.id,
                keep_position=False,
            )
            for executor in self.executors_info
            if executor.is_active
            and self.force_stop_side_required(executor.side)
        ]

    def get_custom_info(self) -> dict:
        now = float(self.market_data_provider.time())
        return {
            "macro_buy_enabled": self.macro_buy_enabled,
            "macro_sell_enabled": self.config.macro_sell_enabled,
            "long_only_enabled": self.config.long_only_enabled,
            "strategy_mode": "LONG_ONLY" if self.config.long_only_enabled else "BILATERAL",
            "sell_creation_enabled": self.sell_creation_enabled,
            "sell_trend_gate": {
                "enabled": self.config.sell_trend_gate_enabled,
                "blocked": self._sell_trend_blocked,
                "reason": self._sell_trend_reason,
                "data_healthy": self._sell_trend_data_healthy,
                "recovery_count": self._sell_trend_recovery_count,
                "recovery_required": self.config.sell_trend_recovery_bars,
                "cooldown_until": self._sell_stop_cooldown_until,
                "cooldown_remaining_seconds": max(0.0, self._sell_stop_cooldown_until - now),
                "stop_event_at": self._sell_stop_event_at,
                "stop_event_id": self._sell_stop_event_id,
                "metrics": dict(self._sell_trend_metrics),
            },
        }
