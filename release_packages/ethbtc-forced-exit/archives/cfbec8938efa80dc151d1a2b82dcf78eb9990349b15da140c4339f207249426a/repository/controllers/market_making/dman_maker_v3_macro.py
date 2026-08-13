from decimal import Decimal
from typing import List

from pydantic import Field, field_validator

from controllers.market_making.dman_maker_v2 import (
    DManMakerV2 as _DManMakerV2,
    DManMakerV2Config as _DManMakerV2Config,
)
from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.models.executor_actions import (
    CreateExecutorAction,
    ExecutorAction,
    StopExecutorAction,
)


class DManMakerV3MacroConfig(_DManMakerV2Config):
    controller_name: str = "dman_maker_v3_macro"
    macro_buy_enabled: bool = Field(
        default=True, json_schema_extra={"is_updatable": True}
    )
    macro_sell_enabled: bool = Field(
        default=True, json_schema_extra={"is_updatable": True}
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


class DManMakerV3Macro(_DManMakerV2):
    """DMan Maker with independently controlled BUY and SELL risk gates."""

    def __init__(self, config: DManMakerV3MacroConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

    @property
    def macro_buy_enabled(self) -> bool:
        return self.config.macro_buy_enabled

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
        if new_config.policy_version != "dca-macro-v3":
            raise ValueError("unsupported macro policy version")
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
        return config

    def create_actions_proposal(self) -> List[ExecutorAction]:
        actions = super().create_actions_proposal()
        return [
            action
            for action in actions
            if not (
                isinstance(action, CreateExecutorAction)
                and (
                    (
                        getattr(action.executor_config, "side", None)
                        == TradeType.BUY
                        and not self.macro_buy_enabled
                    )
                    or (
                        getattr(action.executor_config, "side", None)
                        == TradeType.SELL
                        and not self.config.macro_sell_enabled
                    )
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
            and (
                (executor.side == TradeType.BUY and not self.macro_buy_enabled)
                or (
                    executor.side == TradeType.SELL
                    and not self.config.macro_sell_enabled
                )
            )
        ]
