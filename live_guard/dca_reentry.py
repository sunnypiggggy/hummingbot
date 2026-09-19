"""Persistent DCA reentry; a filled leg stays gated until its cohort commits."""

from contextlib import ExitStack
from copy import deepcopy
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import time
import uuid

try:
    from account_inventory import canonical_sha256
    from emergency_execution import _LeaseHeartbeat, TERMINAL_STATUSES
except ModuleNotFoundError:
    from live_guard.account_inventory import canonical_sha256
    from live_guard.emergency_execution import _LeaseHeartbeat, TERMINAL_STATUSES
from dca_live_common import LIVE_PAIRS, STRATEGY_BUDGET_QUOTE, side_budget, adjustment_timestamp_seconds
from risk_recovery import REENTRY, normalize_state, mark_reentry_complete


class ReentryEvidencePending(RuntimeError):
    """The order is visible before all of its trades have become visible."""


WAIT_REASONS = {
    "insufficient_quote_budget": "重入资金不足，保留已买回库存并等待；普通交易仍关闭",
    "quote_balance_temporarily_unavailable": "余额暂时不可核验，等待后重试",
    "reentry_order_pending_verification": "重入订单结果尚未核验，禁止重复下单",
    "reentry_inventory_pending_verification": "等待重入成交与归属库存对账",
    "reentry_gates_closed": "重入条件或控制器关闭状态尚未满足",
    "reentry_lease_busy": "交易执行租约占用中，等待原执行者完成",
    "reentry_batch_mismatch": "组合恢复批次不一致，等待核验",
    "legacy_reentry_evidence_missing": "旧重入状态缺少成交证据，需人工核验",
    "legacy_reentry_order_unverified": "旧重入订单尚未核验，禁止自动恢复",
    "reentry_evidence_in_wrong_phase": "重入证据与恢复阶段冲突，需人工核验",
    "reentry_exit_waits_for_order_terminal": "保护性退出等待原重入订单终态确认",
    "legacy_reentry_exit_requires_reconciliation": "旧重入证据未核清，保护性退出等待核验",
    "reentry_execution_ledger_unavailable": "重入执行账本不可用，禁止自动下单",
    "reentry_below_exchange_minimum": "重入金额低于交易所下限，等待核验",
    "reentry_existing_orders": "仍有活动订单，暂缓重入",
    "reentry_attempt_limit": "重入补单次数已达上限，需人工核验",
    "reentry_planned_order_price_changed": "已持久化订单不再符合价格或过滤器限制，需核验",
    "reentry_quote_cap_reached": "重入报价币预算已达上限，等待核验",
}


class DcaReentryMixin:
    def _ensure_reentry_batch(self, bot_name):
        state = self.state["bots"][bot_name]["recovery"]
        if state.get("reentry_batch"):
            return state["reentry_batch"]
        names = [bot_name]
        if state.get("scope") == "portfolio":
            names = [spec.bot_name for spec in LIVE_PAIRS.values()]
            peers = [self.state["bots"].get(name, {}).get("recovery", {}) for name in names]
            if any(p.get("scope") != "portfolio" or p.get("phase") != REENTRY
                   or p.get("mechanism") != state.get("mechanism") for p in peers):
                return None
            # Legacy episodes can have slightly different per-leg trigger times.
            # Never attach a new leg to an already assigned (possibly older) batch.
            if any(p.get("reentry_batch") for p in peers):
                return None
        batch = canonical_sha256({
            "bots": {name: {key: self.state["bots"][name]["recovery"].get(key)
                            for key in ("triggered_at", "mechanism", "scope")}
                     for name in names},
        })
        for name in names:
            self.state["bots"][name]["recovery"]["reentry_batch"] = batch
            if len(names) > 1:
                self.state["bots"][name]["recovery"]["reentry_cohort"] = True
        self._save()
        return batch

    def _reentry_wait(self, bot_name, reason):
        state = self.state["bots"][bot_name]["recovery"]
        changed = state.get("reentry_block_reason") != reason
        state["reentry_allowed"] = False
        state["reentry_block_reason"] = reason
        self._save()
        if changed:
            self._audit("recoverable_reentry_wait", bot=bot_name,
                        reason=WAIT_REASONS.get(reason, reason), block_reason=reason,
                        recovery=state)

    def _reentry_permitted(self, state, macro, technical, portfolio_all_gates):
        cohort = state.get("scope") == "portfolio" or state.get("reentry_cohort")
        if cohort:
            peers = [self.state["bots"].get(spec.bot_name, {}) for spec in LIVE_PAIRS.values()]
            if any(peer.get("tripped") or peer.get("recovery", {}).get("phase") != REENTRY
                   for peer in peers):
                return False
            batches = {peer.get("recovery", {}).get("reentry_batch") for peer in peers}
            if len(batches) > 1:
                return False
        return bool(
            self.auto_reentry_enabled and macro.get("healthy")
            and macro.get("buy_enabled") and macro.get("sell_enabled")
            and technical.get("buy_enabled") and technical.get("execution_authorized")
            and not technical.get("force_exit")
            and (not cohort or portfolio_all_gates)
        )

    def _reentry_remaining_quote(self, state):
        entry = state.get("reentry", {})
        if entry.get("status") == "VERIFIED":
            return Decimal("0")
        return max(side_budget() - Decimal(str(entry.get("spent_quote", "0"))), Decimal("0"))

    def _record_reentry_order(self, bot_name, pair, order):
        """Upsert a cumulative exchange order, including partial-fill fees."""
        quantity = Decimal(str(order.get("executedQty", "0")))
        if quantity == 0:
            return self._emergency_fill_metrics(pair, "BUY", order)
        fills = order.get("fills", [])
        unique = {}
        for index, fill in enumerate(fills):
            key = str(fill.get("tradeId", fill.get("id", f"row-{index}")))
            unique[key] = fill
        fills = list(unique.values())
        fill_quantity = sum((Decimal(str(f["qty"])) for f in fills), Decimal("0"))
        if fill_quantity < quantity:
            raise ReentryEvidencePending("reentry trades are not yet complete")
        if fill_quantity != quantity:
            raise ValueError("reentry order fills do not reconcile with executed quantity")
        if not order.get("orderId"):
            raise ValueError("reentry fill has no exchange order id")
        metrics = self._emergency_fill_metrics(pair, "BUY", {**order, "fills": fills})
        rows = self.state["bots"][bot_name].setdefault("emergency_adjustments", [])
        existing = next((r for r in rows if r.get("pair") == pair and r.get("side") == "BUY"
                         and str(r.get("order_id")) == str(order["orderId"])), None)
        if existing and Decimal(existing["executed_qty"]) > quantity:
            raise ValueError("reentry cumulative fill quantity regressed")
        row = {
            "recorded_at": existing["recorded_at"] if existing else datetime.now(timezone.utc).isoformat(),
            "pair": pair, "side": "BUY",
            "order_id": str(order["orderId"]),
            "client_order_id": str(order.get("clientOrderId", "")),
            "executed_qty": str(quantity),
            "cummulative_quote_qty": str(order["cummulativeQuoteQty"]),
            "trade_ids": sorted(unique), **metrics,
        }
        if existing is None:
            rows.append(row)
        else:
            existing.update(row)
        return metrics

    def _settle_reentry_orders(self, bot_name, pair):
        """Read and journal existing orders only. Never retry an uncertain POST."""
        state = self.state["bots"][bot_name]["recovery"]
        entry = state.get("reentry", {})
        ledger = self.inventory_ledger
        terminal = True
        executed = spent = net = fees = Decimal("0")
        fee_details = []
        last_order_id = ""
        for attempt in ledger.attempts(entry["job_id"]):
            response = self.emergency_exchange.order_by_client_id(pair, attempt["client_order_id"])
            if response is None and attempt.get("status") in TERMINAL_STATUSES:
                response = json.loads(attempt["response_json"])
            if response is None:
                if attempt["status"] != "PLANNED":
                    terminal = False
                continue
            if response.get("side") != "BUY" or response.get("symbol") != pair.replace("-", ""):
                raise ValueError("reentry exchange order identity mismatch")
            quantity = Decimal(str(response.get("executedQty", "0")))
            if quantity < 0 or quantity > Decimal(attempt["requested_quantity"]):
                raise ValueError("reentry exchange quantity outside requested bounds")
            try:
                metrics = self._record_reentry_order(bot_name, pair, response)
            except ReentryEvidencePending:
                return False
            executed += quantity
            spent += Decimal(str(response.get("cummulativeQuoteQty", "0")))
            net += Decimal(metrics["base_delta"])
            fees += Decimal(metrics["fee_quote"])
            fee_details.extend(metrics["fee_details"])
            last_order_id = str(response["orderId"])
            status = str(response["status"])
            ledger.finish_attempt(job_id=entry["job_id"], sequence=attempt["sequence"],
                                  status=status, response=response)
            terminal = terminal and status in TERMINAL_STATUSES
        entry.update(executed_qty=str(executed), net_base=str(net), spent_quote=str(spent),
                     fee_quote=str(fees), fee_details=fee_details, last_order_id=last_order_id)
        # Persist accounting before ownership reconciliation on the next cycle.
        self._save()
        return terminal

    def _legacy_reentry(self, bot_name, snapshot):
        """Recover old filled flags only from a unique recorded exchange BUY."""
        state = self.state["bots"][bot_name]["recovery"]
        pair = snapshot["pair"]
        rows = []
        for row in self.state["bots"][bot_name].get("emergency_adjustments", []):
            if row.get("pair") != pair or row.get("side") != "BUY":
                continue
            recorded = row.get("recorded_at", 0)
            recorded = adjustment_timestamp_seconds(recorded)
            if float(recorded) >= float(state.get("exit_completed_at") or float("inf")):
                rows.append(row)
        if len(rows) != 1 or not rows[0].get("order_id"):
            self._reentry_wait(bot_name, "legacy_reentry_evidence_missing")
            return False
        response = self.emergency_exchange.order_by_id(pair, rows[0]["order_id"])
        if not response or not response.get("clientOrderId"):
            self._reentry_wait(bot_name, "legacy_reentry_order_unverified")
            return False
        if (response.get("side") != "BUY" or response.get("symbol") != pair.replace("-", "")
                or response.get("status") != "FILLED"
                or Decimal(str(response.get("executedQty", "0"))) != Decimal(str(
                    state.get("reentry_baseline", {}).get("base", "0")))):
            self._reentry_wait(bot_name, "legacy_reentry_order_unverified")
            return False
        batch = self._ensure_reentry_batch(bot_name)
        if not batch:
            self._reentry_wait(bot_name, "reentry_batch_mismatch")
            return False
        step, _ = self._lot_filter(pair)
        owned, _ = self._verified_owned_base(bot_name, snapshot, step)
        metrics = self._emergency_fill_metrics(pair, "BUY", response)
        job = canonical_sha256({"batch": batch, "bot": bot_name,
                                "triggered_at": state.get("triggered_at")})
        self.inventory_ledger.start_job(
            job_id=job, asset=pair.split("-")[0], scope=f"dca_reentry:{bot_name}",
            pair=pair, requested_quantity=Decimal(response["origQty"]),
            client_order_id=response["clientOrderId"],
        )
        self.inventory_ledger.start_attempt(
            job_id=job, sequence=0, client_order_id=response["clientOrderId"],
            requested_quantity=Decimal(response["origQty"]),
        )
        self.inventory_ledger.finish_attempt(job_id=job, sequence=0,
                                              status=response["status"], response=response)
        state["reentry"] = {
            "schema": "dca-reentry-v1", "batch": batch, "job_id": job,
            "status": "PENDING", "target_quantity": str(response["origQty"]),
            "target_quote": str(side_budget()),
            "starting_owned": str(max(owned - Decimal(metrics["base_delta"]), Decimal("0"))),
            "legacy": True,
        }
        self._save()
        return True

    def _run_reentry(self, bot_name, snapshot, *, allow_submit, now):
        """One bounded exchange step, shared by new, partial and recovered BUYs."""
        state = self.state["bots"][bot_name]["recovery"]
        pair = snapshot["pair"]
        base = pair.split("-")[0]
        ledger = getattr(self, "inventory_ledger", None)
        if ledger is None:
            self._reentry_wait(bot_name, "reentry_execution_ledger_unavailable")
            return
        # Use process-unique holders; a second Guard may not share this lease.
        holder = f"reentry:{os.getpid()}:{uuid.uuid4().hex}"
        with ExitStack() as stack:
            for asset in ("USDT", base):
                if not ledger.acquire_lease(asset, holder, ttl_seconds=45):
                    self._reentry_wait(bot_name, "reentry_lease_busy")
                    return
                stack.callback(ledger.release_lease, asset, holder)
                heartbeat = _LeaseHeartbeat(ledger, asset, holder, 45)
                heartbeat.start()
                stack.callback(heartbeat.stop)
                stack.callback(heartbeat.ensure_owned)
            step, minimum = self._lot_filter(pair)
            mark = Decimal(snapshot["mark_price"])
            counts = self._executor_counts(Path(snapshot["database"]))
            gates = self.state.get("gate_aggregate", {}).get("bots", {}).get(bot_name, {})
            allow_submit = bool(allow_submit and not any(counts.values())
                                and gates.get("controller_applied")
                                and gates.get("controller_actual_buy_enabled") is False
                                and gates.get("controller_actual_sell_enabled") is False)
            if state.get("reentry_filled") and not state.get("reentry"):
                if not self._legacy_reentry(bot_name, snapshot):
                    return
            entry = state.get("reentry", {})
            if not entry:
                if not allow_submit:
                    self._reentry_wait(bot_name, "reentry_gates_closed")
                    return
                batch = self._ensure_reentry_batch(bot_name)
                if not batch:
                    self._reentry_wait(bot_name, "reentry_batch_mismatch")
                    return
                owned, _ = self._verified_owned_base(bot_name, snapshot, step, observed_at=now)
                target = (side_budget() / mark / step).to_integral_value(rounding=ROUND_DOWN) * step
                job = canonical_sha256({"batch": batch, "bot": bot_name,
                                        "triggered_at": state.get("triggered_at")})
                entry = state["reentry"] = {
                    "schema": "dca-reentry-v1", "batch": batch, "job_id": job,
                    "status": "PENDING", "target_quantity": str(target),
                    "target_quote": str(side_budget()), "starting_owned": str(owned),
                }
                self._save()
                ledger.start_job(job_id=job, asset=base, scope=f"dca_reentry:{bot_name}",
                                 pair=pair, requested_quantity=target,
                                 client_order_id=f"dcar-{job[:24]}")
            # Recover the crash window between JSON intent and SQLite job creation.
            if ledger.get_job(entry["job_id"]) is None:
                ledger.start_job(job_id=entry["job_id"], asset=base,
                                 scope=f"dca_reentry:{bot_name}", pair=pair,
                                 requested_quantity=Decimal(entry["target_quantity"]),
                                 client_order_id=f"dcar-{entry['job_id'][:24]}")
            if not self._settle_reentry_orders(bot_name, pair):
                self._reentry_wait(bot_name, "reentry_order_pending_verification")
                return
            remaining = max(Decimal(entry["target_quantity"]) - Decimal(entry["executed_qty"]), Decimal("0"))
            quote_left = max(Decimal(entry["target_quote"]) - Decimal(entry["spent_quote"]), Decimal("0"))
            amount = (min(remaining, quote_left / mark) / step).to_integral_value(rounding=ROUND_DOWN) * step
            # A newly discovered fill must reach the ownership contract before
            # either verification or another economic request can follow it.
            owned, ownership = self._verified_owned_base(bot_name, snapshot, step, observed_at=now)
            expected = Decimal(entry["starting_owned"]) + Decimal(entry["net_base"])
            if abs(owned - expected) > step:
                self._reentry_wait(bot_name, "reentry_inventory_pending_verification")
                return
            if amount <= 0 or amount * mark < minimum:
                if Decimal(entry["executed_qty"]) <= 0:
                    self._reentry_wait(bot_name, "reentry_below_exchange_minimum")
                    return
                if (any(counts.values())
                        or self.emergency_exchange.open_orders(pair)):
                    self._reentry_wait(bot_name, "reentry_inventory_pending_verification")
                    return
                first = entry.get("status") != "VERIFIED"
                entry.update(status="VERIFIED", verified_at=now, ownership=ownership,
                             residual_quantity=str(remaining), residual_quote=str(quote_left))
                ledger.finish_job(
                    entry["job_id"], status="COMPLETED",
                    exchange_order_id=entry["last_order_id"],
                    executed_quantity=entry["executed_qty"], quote_quantity=entry["spent_quote"],
                    fee_quote=entry["fee_quote"], fee_details=entry["fee_details"],
                    verification={"order_verified": True, "balance_verified": True,
                                  "no_active_orders": True, "requested_quantity_verified": True,
                                  "net_base": entry["net_base"], "ownership": ownership,
                                  "residual_quantity": str(remaining), "residual_quote": str(quote_left)},
                )
                state["reentry_filled"] = True
                state["reentry_allowed"] = False
                state["reentry_baseline"] = {"base": entry["net_base"],
                                             "target_quote": entry["target_quote"]}
                state.pop("reentry_block_reason", None)
                self._save()
                if first:
                    self._audit("recoverable_reentry_leg_filled", bot=bot_name, pair=pair,
                                reason="本交易对买回库存已核验；等待恢复提交，普通交易仍关闭",
                                recovery=state)
                return
            if not allow_submit:
                self._reentry_wait(bot_name, "reentry_gates_closed")
                return
            capital = self._quote_budget_status(self._reentry_quote_requirement(
                bot_name, self.state.get("gate_aggregate", {}).get("v22", {})),
                now=now, force_refresh=True)
            if not capital["buy_ready"]:
                state["reentry_capital"] = capital
                self._reentry_wait(bot_name, str(capital["reason"]))
                return
            # Price increases must not turn a fixed base target into extra quote spend.
            quote_left = max(Decimal(entry["target_quote"]) - Decimal(entry["spent_quote"]), Decimal("0"))
            amount = min(amount, (quote_left / mark / step).to_integral_value(rounding=ROUND_DOWN) * step)
            if amount <= 0 or amount * mark < minimum:
                self._reentry_wait(bot_name, "reentry_quote_cap_reached")
                return
            if self.emergency_exchange.open_orders(pair):
                self._reentry_wait(bot_name, "reentry_existing_orders")
                return
            attempts = ledger.attempts(entry["job_id"])
            planned = next((a for a in attempts if a["status"] == "PLANNED"), None)
            sequence = int(planned["sequence"]) if planned else len(attempts)
            if sequence >= 5:
                self._reentry_wait(bot_name, "reentry_attempt_limit")
                return
            client_id = planned["client_order_id"] if planned else f"dcar-{entry['job_id'][:24]}-{sequence}"
            if planned:
                amount = Decimal(planned["requested_quantity"])
                if amount % step or amount * mark > quote_left or amount * mark < minimum:
                    self._reentry_wait(bot_name, "reentry_planned_order_price_changed")
                    return
            else:
                ledger.start_attempt(job_id=entry["job_id"], sequence=sequence,
                                     client_order_id=client_id, requested_quantity=amount)
            # A crash after this marker is deliberately treated as ambiguous,
            # even if the exchange still reports order-not-found.
            ledger.finish_attempt(job_id=entry["job_id"], sequence=sequence, status="SUBMITTING")
            self._save()
            for asset in ("USDT", base):
                if not ledger.renew_lease(asset, holder, ttl_seconds=45):
                    raise RuntimeError("reentry lease lost before submission")
            try:
                response = self.emergency_exchange.market_order(pair, "BUY", amount, client_id)
            except Exception as exc:
                ledger.finish_attempt(job_id=entry["job_id"], sequence=sequence,
                                      status="UNKNOWN", error=repr(exc))
                self._reentry_wait(bot_name, "reentry_order_pending_verification")
                return
            # Journal before any accounting or notification. Subsequent cycles
            # obtain full trades through the order lookup and reconcile ownership.
            ledger.finish_attempt(job_id=entry["job_id"], sequence=sequence,
                                  status=response["status"], response=response)
            self._settle_reentry_orders(bot_name, pair)

    def _commit_reentries(self, snapshots, *, macro, technical, now):
        candidates = []
        portfolio = [spec.bot_name for spec in LIVE_PAIRS.values()]
        states = {name: normalize_state(self.state["bots"].get(name, {}).get("recovery"))
                  for name in snapshots}
        for name, state in states.items():
            if state["scope"] != "portfolio" and not state.get("reentry_cohort"):
                candidates.append([name])
        if all(name in states for name in portfolio):
            peers = [states[n] for n in portfolio]
            if (all(s["scope"] == "portfolio" or s.get("reentry_cohort") for s in peers)
                    and len({s.get("reentry_batch") for s in peers}) == 1
                    and peers[0].get("reentry_batch")):
                candidates.append(portfolio)
        for names in candidates:
            ready = all(
                states[n]["phase"] == REENTRY
                and states[n].get("reentry", {}).get("status") == "VERIFIED"
                and states[n]["reentry"].get("batch") == states[n].get("reentry_batch")
                and states[n]["reentry"].get("verified_at") == now
                and self.state.get("gate_aggregate", {}).get("bots", {}).get(n, {}).get("controller_applied")
                and self.state["gate_aggregate"]["bots"][n].get("controller_actual_buy_enabled") is False
                and self.state["gate_aggregate"]["bots"][n].get("controller_actual_sell_enabled") is False
                and self._reentry_permitted(states[n], macro, technical[snapshots[n]["pair"]], True)
                for n in names
            )
            if not ready:
                continue
            before = deepcopy(self.state)
            for name in names:
                bot = self.state["bots"][name]
                old = bot["recovery"]
                committed = mark_reentry_complete(old, now=now, baseline=old["reentry_baseline"])
                bot.setdefault("reentry_history", []).append(deepcopy(old["reentry"]))
                bot["recovery"] = committed
                # Rebase the recovery risk window together, never the cumulative ledger.
                raw = Decimal(str(snapshots[name].get("raw_pnl_quote", snapshots[name].get("pnl_quote", "0"))))
                bot["pnl_offset_quote"] = str(-raw)
                bot["pnl_offset_pending"] = False
                bot["peak_equity"] = str(STRATEGY_BUDGET_QUOTE)
            if names == portfolio:
                self.state["combined_peak_equity"] = str(STRATEGY_BUDGET_QUOTE * len(portfolio))
            try:
                self._save()
            except Exception:
                self.state = before
                raise
            for name in names:
                self._audit("portfolio_reentry_committed" if names == portfolio else "recoverable_reentry_complete",
                            bot=name, pair=snapshots[name]["pair"], recovery=self.state["bots"][name]["recovery"])

    def _preserve_reentry_for_exit(self, bot_name, replacement):
        previous = self.state["bots"][bot_name].get("recovery", {})
        if previous.get("reentry") or previous.get("reentry_filled"):
            replacement["reentry"] = deepcopy(previous.get("reentry", {}))
            replacement["reentry_batch"] = previous.get("reentry_batch")
            replacement["reentry_cohort"] = bool(previous.get("reentry_cohort") or previous.get("scope") == "portfolio")
            replacement["reentry_abort_pending"] = True
            replacement["legacy_reentry_pending"] = bool(previous.get("reentry_filled") and not previous.get("reentry"))

    def _settle_reentry_before_exit(self, bot_name, pair):
        state = self.state["bots"][bot_name]["recovery"]
        if not state.get("reentry_abort_pending"):
            return True
        if state.get("legacy_reentry_pending"):
            self._reentry_wait(bot_name, "legacy_reentry_exit_requires_reconciliation")
            return False
        if not self._settle_reentry_orders(bot_name, pair):
            self._reentry_wait(bot_name, "reentry_exit_waits_for_order_terminal")
            return False
        self.state["bots"][bot_name].setdefault("reentry_history", []).append(
            {**deepcopy(state["reentry"]), "status": "ABORTED"})
        state["reentry"] = {}
        state.pop("reentry_filled", None)
        state.pop("reentry_baseline", None)
        state.pop("reentry_abort_pending", None)
        self._save()
        return True
