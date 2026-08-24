from hummingbot.strategy_v2.executors.order_executor.data_types import ExecutionStrategy
from hummingbot.strategy_v2.executors.order_executor.order_executor import OrderExecutor
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType, TrackedOrder


class BinanceStocksOrderExecutor(OrderExecutor):
    """Single-attempt Stocks order task.

    LIMIT orders are DAY orders.  An exchange expiry or semantic rejection is a
    terminal task outcome and must not silently create a replacement order on a
    later session.  Unknown HTTP outcomes remain idempotent because the connector
    queries the deterministic client order ID before returning failure.
    """

    def __init__(self, *args, **kwargs):
        kwargs["max_retries"] = 0
        super().__init__(*args, **kwargs)
        if self.config.connector_name != "binance_stocks":
            raise ValueError("BinanceStocksOrderExecutor requires binance_stocks")
        if self.config.execution_strategy not in {ExecutionStrategy.LIMIT, ExecutionStrategy.MARKET}:
            raise ValueError("Binance Stocks OrderExecutor supports LIMIT and MARKET only")

    def paper_checkpoint_state(self):
        return {
            "status": self.status.name,
            "close_type": self.close_type.name if self.close_type else None,
            "order_id": self._order.order_id if self._order else None,
            "current_retries": self._current_retries,
            "held_position_orders": list(self._held_position_orders),
        }

    def restore_paper_checkpoint(self, state):
        order_id = state.get("order_id")
        if order_id:
            tracked = TrackedOrder(order_id=order_id)
            tracked.order = self.get_in_flight_order(self.config.connector_name, order_id)
            self._order = tracked if tracked.order is not None else None
        self._current_retries = int(state.get("current_retries", 0))
        self._held_position_orders = list(state.get("held_position_orders") or [])
        status = str(state.get("status", "RUNNING"))
        self._status = RunnableStatus[status] if status in RunnableStatus.__members__ else RunnableStatus.RUNNING
        close_type = state.get("close_type")
        if close_type and close_type in CloseType.__members__:
            self.close_type = CloseType[close_type]
