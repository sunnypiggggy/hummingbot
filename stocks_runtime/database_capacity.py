"""Low-priority storage upkeep. Never gates economic writes or executors."""
import asyncio
import hashlib
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def latest(fallback=None):
    """Health queries see operator maintenance results without another restart."""
    fallback = fallback or None
    root = Path(os.getenv("STOCK_DATABASE_CAPACITY_ROOT", "/hummingbot-api/logs"))
    try:
        value = json.loads((root/"database_capacity.json").read_text(encoding="utf-8"))
        stamp = value["generated_at"]
        if (not isinstance(stamp, (int, float)) or not math.isfinite(stamp)
                or stamp > time.time()+5 or value.get("level") not in
                {"healthy", "maintenance", "warning", "critical"}
                or value.get("writes_blocked") is not False):
            return fallback
        if fallback and stamp < fallback.get("generated_at", 0):
            return fallback
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return fallback


def publish(root, value, now):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "database_capacity.json"
    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError):
        previous = {}
    episode = previous.get("episode") or hashlib.sha256(str(now).encode()).hexdigest()[:20]
    if previous.get("level") == "healthy" and value["level"] != "healthy":
        episode = hashlib.sha256(str(now).encode()).hexdigest()[:20]
    value = {**value, "generated_at": now, "episode": episode}
    # Stable transition identity survives crashes between append and state replace.
    if value["level"] != previous.get("level") and (value["level"] != "healthy" or previous):
        recovered = value["level"] == "healthy"
        event = {
            "schema": "ethbtc-telegram-event-v1",
            "event_id": hashlib.sha256(f"stock-capacity:{episode}:{value['level']}".encode()).hexdigest(),
            "occurred_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "source": "binance-stocks-runtime", "strategy": "stocks", "bot": "binance-stocks-runtime",
            "pair": "Stock PAPER", "mechanism": "runtime_error",
            "transition": "ERROR_RECOVERED" if recovered else "ERROR_OCCURRED",
            "reason": "Stock数据库容量恢复正常" if recovered else "Stock数据库容量达到治理或告警阈值",
            "severity": "critical" if value["level"] == "critical" else "warning",
            "action": "storage_maintenance_without_blocking_trading",
            "details": {"component": "stock_database_capacity", "error_summary":
                        f"数据库 {value.get('database_bytes', 0)/1048576:.1f} MiB；目标500 MiB；交易写入继续",
                        "trading_impact": "仅存储治理告警，不阻止订单、成交和保护性退出"},
        }
        with (root / "telegram_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
    temp = root / ".database_capacity.tmp"
    temp.write_text(json.dumps(value), encoding="utf-8")
    os.replace(temp, path)
    return value


async def run(app, stop):
    broker = app.state.stocks_paper_broker
    while not stop.is_set():
        try:
            async with broker.ledger._pool.acquire() as connection:
                async with connection.transaction():
                    await connection.execute("SET LOCAL lock_timeout='2s'")
                    await connection.execute("SET LOCAL statement_timeout='15s'")
                    value = await broker.quote_store.maintenance(connection)
            app.state.stocks_database_capacity = publish(
                os.getenv("STOCK_DATABASE_CAPACITY_ROOT", "/hummingbot-api/logs"), value, time.time())
        except Exception:
            logging.getLogger(__name__).exception("Stock storage upkeep failed; trading writes remain enabled")
        try:
            await asyncio.wait_for(stop.wait(), timeout=3600)
        except asyncio.TimeoutError:
            pass
