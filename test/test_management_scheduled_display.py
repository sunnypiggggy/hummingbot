import time
from unittest.mock import Mock

from management_bot.app import TradingManagementBot
from management_bot.clients import ServiceError
from management_bot.scheduled_display import detail, fmt


def order(i=1, **extra):
    return dict(schedule_id=f"sch-{i:024d}", request_type="position_executor",
                request_payload={"symbol": "AAPL", "entry_order_type": "LIMIT",
                                 "entry_price": "100", "take_profit": "0.03",
                                 "stop_loss": "0.02", "time_limit": 604800},
                frozen_price="100", requested_shares="2", status="WAITING_SESSION",
                **extra)


def render(item, **kwargs):
    return detail(item, TradingManagementBot._scheduled_status_cn,
                  TradingManagementBot._target_session_cn, **kwargs)


def bot(item=None):
    b = object.__new__(TradingManagementBot)
    b.stocks = Mock()
    b.stocks.scheduled_detail.return_value = item or order()
    b.stocks.health.return_value = {"runtime_mode": "PAPER"}
    b._stock_quote_context = Mock(side_effect=RuntimeError("offline"))
    b.store = Mock()
    return b


def test_saved_limit_barriers_survive_quote_failure():
    b = bot()
    text, rows = b._stock_scheduled_detail(order()["schedule_id"])
    assert "103 USDC" in text and "98 USDC" in text
    assert "最长持仓：7 天" in text and "委托价：100 USDC" in text
    assert "PAPER" in text and "行情查询失败" in text
    assert len(text) < 4096
    b.stocks.refresh_scheduled.assert_not_called()
    b.stocks.cancel_scheduled.assert_not_called()


def test_missing_and_unset_barriers_never_gain_defaults():
    item = order()
    item["request_payload"] = {"symbol": "AAPL", "stop_loss": None}
    text = render(item)
    assert "止盈：未记录" in text and "止损：未设置" in text
    assert "最长持仓：未记录" in text
    item["request_type"] = "order_executor"
    assert "单笔订单不附带仓位止盈止损" in render(item)


def test_market_budget_stale_quote_does_not_invent_entry_or_barriers():
    item = order(quote_budget="100")
    item["request_payload"]["entry_order_type"] = "MARKET"
    text = render(item, quote={"quote_ts": time.time() - 600, "reference": "120"})
    assert "开市按当时行情执行" in text and "固定预算：100 USDC" in text
    assert "触发参考价" not in text and "委托价：" not in text
    fresh = render(item, quote={"quote_ts": time.time(), "reference": "120"})
    assert "123.6 USDC" in fresh and "117.6 USDC" in fresh
    assert "成交后确定实际触发价" in fresh
    assert fmt(0) == "0"


def test_same_stock_pagination_binds_full_ids():
    b = bot()
    b.stocks.scheduled.return_value = [order(i) for i in range(13)]
    callbacks, labels = [], []
    for page in range(3):
        text, rows = b._stock_scheduled_menu(page)
        entries = [(label, cb) for row in rows for label, cb in row if cb.startswith("q:")]
        assert len(entries) == (6 if page < 2 else 1)
        labels += [label for label, _ in entries]
        callbacks += [cb for _, cb in entries]
        assert all("cancel" not in cb for _, cb in entries)
        assert all(len(cb.encode()) <= 64 for row in rows for _, cb in row)
    assert len(set(labels)) == len(set(callbacks)) == 13


def test_cancel_preview_no_write_and_activation_race_no_false_success():
    b = bot()
    sid = order()["schedule_id"]
    preview, rows = b._stock_scheduled_cancel(sid)
    assert "AAPL" in preview and "确认撤销" in preview
    b.stocks.cancel_scheduled.assert_not_called()
    b.stocks.cancel_scheduled.return_value = {"schedule": {**order(), "status": "ACTIVE"}, "executor_active": True}
    text, _ = b._stock_scheduled_cancel(sid, confirmed=True)
    assert "本次未撤销" in text and "已释放" not in text
    b.stocks.cancel_scheduled.assert_called_once_with(sid)


def test_terminal_old_button_does_not_cancel_again():
    item = order()
    item["status"] = "CANCELED"
    b = bot(item)
    _, rows = b._stock_scheduled_cancel(item["schedule_id"], confirmed=True)
    b.stocks.cancel_scheduled.assert_not_called()
    assert not any("cancel" in cb for row in rows for _, cb in row)


def test_real_http_409_queries_active_executor():
    b = bot()
    active = {**order(), "status": "ACTIVE"}
    b.stocks.scheduled_detail.side_effect = [order(), active, active]
    b.stocks.cancel_scheduled.side_effect = ServiceError("already active", 409)
    text, rows = b._stock_scheduled_cancel(order()["schedule_id"], confirmed=True)
    assert "本次未撤销" in text and "已释放" not in text
    assert any(cb == "s:positions" for row in rows for _, cb in row)
