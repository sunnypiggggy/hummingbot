"""Isolated harness using the production recovery, journal and ownership check."""
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "live_guard"), str(ROOT / "scripts")]
from account_inventory import UnifiedInventoryLedger
from dca_live_guard import Guard, BinanceEmergencyClient
from dca_live_common import LIVE_PAIRS

MACRO = {"healthy": True, "buy_enabled": True, "sell_enabled": True}
TECHNICAL = {p: {"buy_enabled": True, "execution_authorized": True,
                 "force_exit": False} for p in LIVE_PAIRS}


def make_guard(root, exchange, *, load=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    guard = Guard.__new__(Guard)
    guard.state_path = root / "guard.json"
    guard.inventory_ledger = UnifiedInventoryLedger(root / "inventory")
    guard.emergency_exchange = exchange
    guard.auto_reentry_enabled = True
    guard.quote_budget_buffer_pct = Decimal("0.002")
    guard.quote_balance_cache_seconds = 30
    guard._lot_filter = lambda pair: (Decimal("0.00001" if pair.startswith("BTC") else "0.0001"), Decimal("5"))
    guard._executor_counts = lambda path: {"open_orders": 0, "active_buy_executors": 0}
    guard.audit_events = []
    guard._audit = lambda event, **kw: guard.audit_events.append((event, kw))
    if load:
        guard.state = json.loads(guard.state_path.read_text())
    else:
        guard.state = {"bots": {}, "gate_aggregate": {"v22": {"pairs": TECHNICAL}, "bots": {}}}
        for pair, spec in LIVE_PAIRS.items():
            guard.state["bots"][spec.bot_name] = {
                "managed_base_target": "1", "recovery": {
                    "phase": "REENTRY", "scope": "portfolio",
                    "mechanism": "portfolio_drawdown_breaker", "triggered_at": 1,
                    "exit_completed_at": 2, "healthy_cycles": 3,
                },
            }
            guard.state["gate_aggregate"]["bots"][spec.bot_name] = {
                "controller_applied": True, "controller_actual_buy_enabled": False,
                "controller_actual_sell_enabled": False,
            }
        guard._save()
    return guard


def snapshots(guard, prices=None):
    prices = prices or {"BTC-USDT": "77000", "ETH-USDT": "2500"}
    result, owners = {}, {}
    for pair, spec in LIVE_PAIRS.items():
        rows = guard.state["bots"][spec.bot_name].get("emergency_adjustments", [])
        owned = sum((Decimal(r["base_delta"]) for r in rows), Decimal("0"))
        result[spec.bot_name] = {"pair": pair, "mark_price": prices[pair], "database": "unused",
                                 "net_base": str(owned - 1), "pnl_quote": "7", "raw_pnl_quote": "7"}
        owners[pair.split("-")[0]] = {f"dca:{spec.bot_name}": str(owned)}
    guard.inventory_ledger.reconcile(
        account_fingerprint="reentry-scenario", balances=guard.emergency_exchange.account_balances(),
        ownership=owners, evidence_sha256="isolated", open_order_counts={},
        sources_healthy=True, now=time.time(),
    )
    return result


def tick(guard, *, order=None, macro=None, technical=None, omit=None):
    current = snapshots(guard)
    now = time.time()
    if omit:
        current.pop(omit)
    macro, technical = macro or MACRO, technical or TECHNICAL
    for name in order or current:
        if name not in current:
            continue
        snapshot = current[name]
        guard._process_recoverable(name, snapshot, macro=macro,
                                   technical=technical[snapshot["pair"]], now=now,
                                   portfolio_all_gates=all(v["buy_enabled"] for v in technical.values()))
    guard._commit_reentries(current, macro=macro, technical=technical, now=now)
    guard._save()


if __name__ == "__main__":
    # Used by the process-crash acceptance test with a loopback-only fake exchange.
    exchange = BinanceEmergencyClient("scenario-key", "scenario-secret", sys.argv[2])
    guard = make_guard(sys.argv[1], exchange, load=Path(sys.argv[1], "guard.json").exists())
    if len(sys.argv) > 3 and sys.argv[3] == "crash_after_fill":
        import os
        original = exchange.market_order
        def crash(*args, **kwargs):
            original(*args, **kwargs)
            os._exit(73)
        exchange.market_order = crash
    for _ in range(4):
        tick(guard)
