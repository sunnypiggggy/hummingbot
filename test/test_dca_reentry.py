import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from decimal import Decimal

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test" / "support"))
from dca_reentry_harness import make_guard, snapshots, tick, MACRO, TECHNICAL, LIVE_PAIRS
from risk_scenario_server import RiskScenarioServer
from dca_live_guard import BinanceEmergencyClient

NAMES = [s.bot_name for s in LIVE_PAIRS.values()]


class Exchange:
    def __init__(self):
        self.cash = Decimal("450")
        self.base = {"BTC": Decimal("0"), "ETH": Decimal("0")}
        self.orders = {}
        self.posts = []
        self.fraction = Decimal("1")
        self.status = "FILLED"
        self.drop = False
        self.invisible = False

    def account_balances(self):
        return {asset: {"free": value, "locked": Decimal("0"), "total": value}
                for asset, value in {**self.base, "USDT": self.cash}.items()}

    def open_orders(self, pair):
        return []

    def cancel_all_orders(self, pair):
        raise AssertionError("reentry must not cancel or sell its completed leg")

    def market_order(self, pair, side, amount, client_id):
        assert side == "BUY"
        assert client_id not in self.orders
        price = Decimal("77000" if pair.startswith("BTC") else "2500")
        qty = amount * self.fraction
        fee = qty * Decimal("0.001")
        self.cash -= qty * price
        self.base[pair.split("-")[0]] += qty - fee
        row = {"symbol": pair.replace("-", ""), "side": side, "orderId": len(self.posts) + 1,
               "clientOrderId": client_id, "status": self.status, "origQty": str(amount),
               "executedQty": str(qty), "cummulativeQuoteQty": str(qty * price),
               "fills": [{"tradeId": len(self.posts) + 1, "qty": str(qty), "price": str(price),
                          "commission": str(fee), "commissionAsset": pair.split("-")[0]}]}
        self.posts.append(row)
        self.orders[client_id] = row
        if self.drop:
            raise TimeoutError("response lost after fill")
        return row

    def order_by_client_id(self, pair, client_id):
        return None if self.invisible else self.orders.get(client_id)

    def order_by_id(self, pair, order_id):
        return next(v for v in self.orders.values() if str(v["orderId"]) == str(order_id))


@pytest.fixture
def pair(tmp_path):
    exchange = Exchange()
    return make_guard(tmp_path, exchange), exchange


@pytest.mark.parametrize("reverse", [False, True])
def test_sequential_legs_wait_then_commit_once(pair, reverse):
    guard, exchange = pair
    order = NAMES[::-1] if reverse else NAMES
    tick(guard, order=[order[0]])
    assert len(exchange.posts) == 1
    assert guard._reentry_quote_requirement(order[1], {"pairs": TECHNICAL}) < Decimal("287")
    # Force the second leg to wait several cycles, retaining the first purchase.
    saved = exchange.cash
    exchange.cash = Decimal("200")
    for _ in range(3):
        tick(guard, order=order)
    assert len(exchange.posts) == 1
    assert all(b["recovery"]["phase"] == "REENTRY" for b in guard.state["bots"].values())
    exchange.cash = saved
    tick(guard, order=order)
    tick(guard, order=order)
    tick(guard, order=order)
    assert len(exchange.posts) == 2
    assert all(b["recovery"]["phase"] == "ACTIVE" for b in guard.state["bots"].values())
    assert sum(e == "portfolio_reentry_committed" for e, _ in guard.audit_events) == 2
    assert sum(len(b["emergency_adjustments"]) for b in guard.state["bots"].values()) == 2


def test_same_cycle_fee_net_inventory_and_cumulative_accounting(pair):
    guard, exchange = pair
    tick(guard)
    tick(guard)
    for name, bot in guard.state["bots"].items():
        assert bot["recovery"]["phase"] == "ACTIVE"
        row = bot["emergency_adjustments"][0]
        assert Decimal(bot["recovery"]["episode_baseline"]["base"]) == Decimal(row["executed_qty"]) * Decimal("0.999")
        assert Decimal(row["fee_quote"]) > 0
        assert bot["pnl_offset_quote"] == "-7"


@pytest.mark.parametrize("change", ["missing", "batch", "EXITING", "COOLDOWN"])
def test_invalid_cohort_never_commits(pair, change):
    guard, exchange = pair
    tick(guard)
    if change == "missing":
        tick(guard, omit=NAMES[1])
    elif change == "batch":
        guard.state["bots"][NAMES[1]]["recovery"]["reentry_batch"] = "other"
        tick(guard)
    else:
        guard.state["bots"][NAMES[1]]["recovery"]["phase"] = change
        current = snapshots(guard)
        guard._commit_reentries(current, macro=MACRO, technical=TECHNICAL, now=time.time())
    assert all(b["recovery"]["phase"] != "ACTIVE" for b in guard.state["bots"].values())
    assert len(exchange.posts) == 2


@pytest.mark.parametrize("gate", ["fomc", "authorization", "risk_off", "controller"])
def test_pending_leg_does_not_submit_or_commit_with_closed_gate(pair, gate):
    guard, exchange = pair
    tick(guard, order=[NAMES[0]])
    macro, technical = copy.deepcopy(MACRO), copy.deepcopy(TECHNICAL)
    if gate == "fomc":
        macro["buy_enabled"] = False
    elif gate == "controller":
        guard.state["gate_aggregate"]["bots"][NAMES[1]]["controller_applied"] = False
    else:
        for t in technical.values():
            t["execution_authorized" if gate == "authorization" else "buy_enabled"] = False
    tick(guard, macro=macro, technical=technical)
    assert len(exchange.posts) == 1
    assert all(b["recovery"]["phase"] == "REENTRY" for b in guard.state["bots"].values())


def test_timeout_invisible_then_restart_recovers_exact_order(pair, tmp_path):
    guard, exchange = pair
    exchange.drop = True
    tick(guard, order=[NAMES[0]])
    exchange.invisible = True
    for _ in range(3):
        guard = make_guard(tmp_path, exchange, load=True)
        tick(guard, order=[NAMES[0]])
    assert len(exchange.posts) == 1
    exchange.invisible = exchange.drop = False
    for _ in range(3):
        tick(guard)
    assert len(exchange.posts) == 2
    assert all(b["recovery"]["phase"] == "ACTIVE" for b in guard.state["bots"].values())


def test_terminal_partial_only_buys_remaining_and_deduplicates(pair):
    guard, exchange = pair
    exchange.fraction, exchange.status = Decimal("0.5"), "EXPIRED"
    tick(guard, order=[NAMES[0]])
    first = exchange.posts[0]
    exchange.fraction, exchange.status = Decimal("1"), "FILLED"
    tick(guard, order=[NAMES[0]])
    tick(guard, order=[NAMES[0]])
    assert len(exchange.posts) == 2
    residual = Decimal(first["origQty"]) - sum(Decimal(o["executedQty"]) for o in exchange.posts)
    assert Decimal("0") <= residual < Decimal("0.00001")
    tick(guard, order=[NAMES[0]])
    assert len(guard.state["bots"][NAMES[0]]["emergency_adjustments"]) == 2
    assert len(exchange.posts) == 2


def test_nonterminal_partial_blocks_children(pair):
    guard, exchange = pair
    exchange.fraction, exchange.status = Decimal("0.5"), "PARTIALLY_FILLED"
    for _ in range(3):
        tick(guard, order=[NAMES[0]])
    assert len(exchange.posts) == 1


def test_duplicate_trade_rows_do_not_duplicate_fees(pair):
    guard, exchange = pair
    tick(guard, order=[NAMES[0]])
    order = exchange.posts[0]
    order["fills"].append(copy.deepcopy(order["fills"][0]))
    tick(guard, order=[NAMES[0]])
    entry = guard.state["bots"][NAMES[0]]["recovery"]["reentry"]
    assert entry["status"] == "VERIFIED"
    assert Decimal(entry["net_base"]) == Decimal(order["executedQty"]) * Decimal("0.999")


def test_trade_visibility_delay_is_wait_not_guard_error(pair):
    guard, exchange = pair
    tick(guard, order=[NAMES[0]])
    fills = exchange.posts[0]["fills"]
    exchange.posts[0]["fills"] = []
    tick(guard, order=[NAMES[0]])
    assert len(exchange.posts) == 1
    exchange.posts[0]["fills"] = fills
    tick(guard, order=[NAMES[0]])
    assert guard.state["bots"][NAMES[0]]["recovery"]["reentry"]["status"] == "VERIFIED"


def test_proven_legacy_fills_migrate_without_repeat_purchase(pair):
    guard, exchange = pair
    tick(guard)
    for name in NAMES:
        state = guard.state["bots"][name]["recovery"]
        state["reentry_filled"] = True
        state["reentry_baseline"] = {"base": state["reentry"]["executed_qty"]}
        state["reentry"] = {}
        state.pop("reentry_batch")
    tick(guard)
    assert len(exchange.posts) == 2
    assert all(b["recovery"]["phase"] == "ACTIVE" for b in guard.state["bots"].values())


def test_complete_guard_cycle_keeps_gates_closed_until_commit(pair):
    from types import SimpleNamespace
    guard, exchange = pair
    current = {}
    guard.api = SimpleNamespace(status=lambda: NAMES)
    guard._observe_v22_contract = lambda now: None
    def reconcile(**kwargs):
        current.clear()
        current.update(snapshots(guard))
    guard._reconcile_account_inventory = reconcile
    guard._snapshot = lambda name, pair: dict(current[name])
    guard._consume_position_stop_event = lambda bot, snap: (None, False)
    guard._macro_gate = lambda: {**MACRO, "reason": "clear"}
    technical = {p: {**v, "reason": "clear", "source_pair": p} for p, v in TECHNICAL.items()}
    guard._v21_gate = lambda: {"healthy": True, "pairs": technical}
    guard._clear_integrity_failure = lambda *args: None
    applied = []
    def set_gates(name, snapshot, *, buy_enabled, sell_enabled, reasons):
        applied.append((name, buy_enabled, sell_enabled))
        return {"status": "applied", "macro_buy_enabled": buy_enabled,
                "macro_sell_enabled": sell_enabled}
    guard._set_effective_gates = set_gates
    guard._record_read_retry_events = lambda: None
    guard._write_macro_telemetry = lambda snapshots: None
    for _ in range(2):
        guard.cycle()
    assert all(not buy and not sell for _, buy, sell in applied)
    assert all(b["recovery"]["phase"] == "ACTIVE" for b in guard.state["bots"].values())
    guard.cycle()
    assert all(buy and sell for _, buy, sell in applied[-2:])
    assert len(exchange.posts) == 2


def test_lease_contention_never_submits(pair):
    guard, exchange = pair
    guard.inventory_ledger.acquire_lease("USDT", "other-process", ttl_seconds=60)
    tick(guard)
    assert exchange.posts == []


def test_commit_disk_failure_rolls_back_in_memory_and_restart(pair, tmp_path):
    guard, exchange = pair
    tick(guard)
    current = snapshots(guard)
    now = time.time()
    for name in NAMES:
        guard._process_recoverable(name, current[name], macro=MACRO,
                                   technical=TECHNICAL[current[name]["pair"]], now=now)
    save = guard._save
    guard._save = lambda: (_ for _ in ()).throw(OSError("disk unavailable"))
    with pytest.raises(OSError):
        guard._commit_reentries(current, macro=MACRO, technical=TECHNICAL, now=now)
    assert all(b["recovery"]["phase"] == "REENTRY" for b in guard.state["bots"].values())
    guard._save = save
    guard = make_guard(tmp_path, exchange, load=True)
    tick(guard)
    assert len(exchange.posts) == 2
    assert all(b["recovery"]["phase"] == "ACTIVE" for b in guard.state["bots"].values())


def test_risk_exit_retains_journal_and_settles_ambiguous_buy(pair):
    guard, exchange = pair
    exchange.drop = True
    tick(guard, order=[NAMES[0]])
    current = snapshots(guard)
    guard._trigger_recoverable(NAMES[0], current[NAMES[0]], mechanism="v22_weekly_buy_gate",
                               scope="technical", trigger_value="off", reason="risk_off")
    recovery = guard.state["bots"][NAMES[0]]["recovery"]
    assert recovery["phase"] == "EXITING" and recovery["reentry_abort_pending"]
    exchange.invisible = True
    assert not guard._settle_reentry_before_exit(NAMES[0], "BTC-USDT")
    exchange.invisible = False
    assert guard._settle_reentry_before_exit(NAMES[0], "BTC-USDT")
    assert not recovery.get("reentry_filled")
    assert len(guard.state["bots"][NAMES[0]]["emergency_adjustments"]) == 1
    assert len(exchange.posts) == 1


@pytest.mark.parametrize("phase", ["LATCHED", "COOLDOWN", "REENTRY", "EXITING"])
def test_unproven_legacy_flag_cannot_create_trades(pair, phase):
    guard, exchange = pair
    for b in guard.state["bots"].values():
        b["recovery"].update(phase=phase, reentry_filled=True)
    before = copy.deepcopy(guard.state["bots"])
    tick(guard)
    assert not exchange.posts
    assert all(b["recovery"]["phase"] == phase for b in guard.state["bots"].values())
    if phase == "LATCHED":
        assert before == guard.state["bots"]


def scenario_document():
    return {"balances": {"USDT": {"free": "450"}, "BTC": {"free": "0"}, "ETH": {"free": "0"}},
            "prices": {"BTCUSDT": "77000", "ETHUSDT": "2500"},
            "filters": {"BTCUSDT": {"lot_step": "0.00001", "min_notional": "5"},
                        "ETHUSDT": {"lot_step": "0.0001", "min_notional": "5"}}, "faults": {}}


@pytest.mark.parametrize("crash", [False, True])
def test_http_response_loss_and_process_restart(tmp_path, crash):
    doc = scenario_document()
    if not crash:
        doc["faults"] = {"POST /api/v3/order": [{"drop_response": True, "visibility_misses": 2,
                                                   "commission": "0.00000123", "commission_asset": "BTC"}]}
    with RiskScenarioServer(doc) as server:
        env = {**os.environ, "GUARD_SCENARIO_MODE": "true", "GUARD_SCENARIO_ID": "dca-reentry",
               "BINANCE_API_BASE_URL": server.base_url}
        args = [sys.executable, str(ROOT / "test/support/dca_reentry_harness.py"),
                str(tmp_path), server.base_url]
        if crash:
            first = subprocess.run(args + ["crash_after_fill"], env=env, capture_output=True, timeout=25)
            assert first.returncode == 73, first.stderr.decode()
            # Another process must wait for the killed holder's real lease expiry.
            early = subprocess.run(args, env=env, capture_output=True, timeout=25)
            assert early.returncode == 0, early.stderr.decode()
            assert len(server.state.orders) == 1
            time.sleep(46)
        for _ in range(2):
            run = subprocess.run(args, env=env, capture_output=True, timeout=25)
            assert run.returncode == 0, run.stderr.decode()
        state = json.loads((tmp_path / "guard.json").read_text())
        assert all(b["recovery"]["phase"] == "ACTIVE" for b in state["bots"].values())
        remote = server.server.scenario_state.public()
        assert len(remote["orders"]) == 2
        assert all(o["side"] == "BUY" for o in remote["orders"])
        assert sum(len(b["emergency_adjustments"]) for b in state["bots"].values()) == 2


@pytest.mark.parametrize("historical", [None, 1789523935.294364, "1789523935.758684"])
def test_reentry_timestamp_preserved_through_partial_update_and_report(pair, historical):
    from datetime import datetime, timezone
    from live_guard.dca_live_report import calculate_pair_report
    guard, exchange = pair
    order = exchange.market_order("BTC-USDT", "BUY", Decimal("0.001"), "time-test")
    guard._record_reentry_order(NAMES[0], "BTC-USDT", order)
    rows = guard.state["bots"][NAMES[0]]["emergency_adjustments"]
    assert datetime.fromisoformat(rows[0]["recorded_at"]).utcoffset().total_seconds() == 0
    if historical is not None:
        rows[0]["recorded_at"] = historical
    original = rows[0]["recorded_at"]
    larger = copy.deepcopy(order)
    for key in ("executedQty", "cummulativeQuoteQty"):
        larger[key] = str(Decimal(larger[key]) * 2)
    for key in ("qty", "commission"):
        larger["fills"][0][key] = str(Decimal(larger["fills"][0][key]) * 2)
    guard._record_reentry_order(NAMES[0], "BTC-USDT", larger)
    guard._record_reentry_order(NAMES[0], "BTC-USDT", larger)
    assert len(rows) == 1 and rows[0]["recorded_at"] == original
    assert Decimal(rows[0]["executed_qty"]) == Decimal("0.002")
    now = datetime.now(timezone.utc)
    report = calculate_pair_report(bot_name=NAMES[0], pair="BTC-USDT", rows=[],
        candles=[{"timestamp": now.timestamp(), "close": "77000"}],
        database_age_seconds=0, now=now, emergency_adjustments=rows)
    assert Decimal(report["position"]["net_base"]) == Decimal("0.001998")
