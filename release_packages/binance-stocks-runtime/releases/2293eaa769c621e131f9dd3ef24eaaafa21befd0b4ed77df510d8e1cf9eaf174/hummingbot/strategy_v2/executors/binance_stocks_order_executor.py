from hummingbot.strategy_v2.executors.order_executor.data_types import ExecutionStrategy
from hummingbot.strategy_v2.executors.order_executor.order_executor import OrderExecutor


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
