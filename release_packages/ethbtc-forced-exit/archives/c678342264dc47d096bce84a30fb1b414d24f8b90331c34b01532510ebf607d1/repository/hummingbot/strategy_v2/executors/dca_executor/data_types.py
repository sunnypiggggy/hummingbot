from decimal import Decimal
from enum import Enum
from typing import List, Literal, Optional

from pydantic import Field

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.executors.data_types import ExecutorConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop


class DCAMode(Enum):
    MAKER = "MAKER"
    TAKER = "TAKER"


class DCAExecutorConfig(ExecutorConfigBase):
    type: Literal["dca_executor"] = "dca_executor"
    connector_name: str
    trading_pair: str
    side: TradeType
    leverage: int = 1
    amounts_quote: List[Decimal]
    prices: List[Decimal]
    take_profit: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    trailing_stop: Optional[TrailingStop] = None
    time_limit: Optional[int] = None
    # Optional safer semantics for long-lived maker ladders. When enabled, an
    # unfilled ladder is refreshed by its controller, while the position time
    # limit starts at the first fill instead of executor creation.
    time_limit_from_first_fill: bool = False
    # Preserve the legacy maker behavior by default. Live macro DCA enables
    # this so a partially-filled ladder is protected by stop loss immediately.
    stop_loss_on_partial_fills: bool = False
    # Poll shutdown completion frequently enough that a failed/partial close
    # cannot leave risk unmanaged for an entire strategy tick. Controllers may
    # tighten this value, while validation prevents a busy loop or a long gap.
    shutdown_retry_seconds: float = Field(default=1.0, ge=0.2, le=5.0)
    mode: DCAMode = DCAMode.MAKER
    activation_bounds: Optional[List[Decimal]] = None
    level_id: Optional[str] = None
