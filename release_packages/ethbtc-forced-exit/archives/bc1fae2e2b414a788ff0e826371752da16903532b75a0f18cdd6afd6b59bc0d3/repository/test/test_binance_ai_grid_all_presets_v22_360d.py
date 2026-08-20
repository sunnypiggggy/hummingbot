from scripts.backtest_binance_ai_grid_all_presets_v22_360d import AI_PRESETS, SOURCE_LEVELS, returns_matrix
from scripts.backtest_binance_short_sideways_long_vs_bidirectional_360d import (
    MINIMUM_ORDER,
    ExchangeFilter,
    build_grid_orders,
)


def test_all_four_binance_ai_presets_use_18_executable_levels():
    assert list(AI_PRESETS) == ["short_sideways", "medium_sideways", "medium_volatility", "long_volatility"]
    assert SOURCE_LEVELS == {"short_sideways": 25, "medium_sideways": 42, "medium_volatility": 28, "long_volatility": 80}
    assert all(preset.levels == 18 and preset.side_levels == 9 for preset in AI_PRESETS.values())


def test_all_presets_respect_10_fdusd_after_eth_quantisation():
    exchange_filter = ExchangeFilter(tick_size=0.01, step_size=0.0001, minimum_notional=5.0)
    for preset in AI_PRESETS.values():
        orders, _ = build_grid_orders(pair="ETH-FDUSD", mode="bidirectional", preset=preset,
                                      exchange_filter=exchange_filter, center=4500.0,
                                      quote_available=100.0, base_available=100.0 / 4500.0, bar=0)
        assert orders
        assert all(order.notional + 1e-9 >= MINIMUM_ORDER for order in orders)
        assert len([order for order in orders if order.side == "BUY"]) <= 9
        assert len([order for order in orders if order.side == "SELL"]) <= 9
