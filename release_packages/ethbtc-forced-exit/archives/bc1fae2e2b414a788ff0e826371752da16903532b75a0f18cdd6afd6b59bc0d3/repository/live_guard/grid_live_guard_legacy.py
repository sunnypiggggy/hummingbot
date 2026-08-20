#!/usr/bin/env python3
"""External fail-closed guard for the isolated live Grid portfolios."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from grid_live_common import PORTFOLIO_DRAWDOWN_LIMIT_PCT, PORTFOLIOS, budget_for_quote
from grid_technical_gate import (
    COMBINED_RECOVERY_RULE_VERSION,
    atomic_json as atomic_gate_json,
    build_technical_buy_gate,
    failed_technical_buy_gate,
    roc_sqz_signal_from_klines,
)

try:
    from live_guard.dca_live_guard import BinanceEmergencyClient, DockerEmergencyClient
except ImportError:  # Docker image layout
    from dca_live_guard import BinanceEmergencyClient, DockerEmergencyClient


LOG = logging.getLogger("grid-live-guard")
BINANCE_API = "https://api.binance.com"
SCALE = Decimal("1000000")


def fill_pnl(rows: list[tuple[Any, ...]], mark_price: Decimal) -> tuple[Decimal, Decimal]:
    cashflow, net_base, fees = Decimal("0"), Decimal("0"), Decimal("0")
    for side, price_raw, amount_raw, fee_raw in rows:
        price, amount = Decimal(price_raw) / SCALE, Decimal(amount_raw) / SCALE
        fee = Decimal(fee_raw or 0) / SCALE
        if str(side).upper() == "BUY":
            cashflow -= price * amount
            net_base += amount
        elif str(side).upper() == "SELL":
            cashflow += price * amount
            net_base -= amount
        fees += fee
    return cashflow + net_base * mark_price - fees, net_base


def peak_drawdown(current_equity: Decimal, stored_peak: Decimal,
                  initial_equity: Decimal) -> tuple[Decimal, Decimal]:
    peak = max(stored_peak, initial_equity, current_equity)
    drawdown = (peak - current_equity) / peak if peak > 0 else Decimal("0")
    return peak, drawdown


class ApiClient:
    def __init__(self):
        self.base = os.getenv("HUMMINGBOT_API_URL", "http://hummingbot-api:8000").rstrip("/")
        self.session = requests.Session()
        self.session.auth = (os.environ["USERNAME"], os.environ["PASSWORD"])

    def request(self, method: str, path: str, payload: Dict[str, Any] | None = None) -> Any:
        response = self.session.request(method, f"{self.base}{path}", json=payload, timeout=30)
        response.raise_for_status()
        return response.json() if response.content else {}

    def status(self) -> Any:
        return self.request("GET", "/bot-orchestration/status")

    def stop(self, bot_name: str) -> Any:
        return self.request("POST", "/bot-orchestration/stop-bot", {
            "bot_name": bot_name, "skip_order_cancellation": False, "async_backend": False,
        })

    def active_containers(self, name_filter: str) -> Any:
        response = self.session.get(
            f"{self.base}/docker/active-containers",
            params={"name_filter": name_filter},
            timeout=30,
        )
        response.raise_for_status()
        return response.json() if response.content else []

    def market(self, profile: str, pair: str, side: str, amount: Decimal) -> Any:
        return self.request("POST", "/trading/orders", {
            "account_name": profile, "connector_name": "binance", "trading_pair": pair,
            "trade_type": side, "amount": str(amount), "order_type": "MARKET", "position_action": "OPEN",
        })


class Guard:
    def __init__(self):
        self.bots_path = Path(os.getenv("BOTS_PATH", "/workspace/bots"))
        self.state_dir = Path(os.getenv("GRID_LIVE_STATE_PATH", "/workspace/grid-live-state"))
        self.manifest_path = Path(os.getenv(
            "GRID_LIVE_RESERVATIONS", "/workspace/grid-live-state/capital_reservations.json"
        ))
        self.manifest = (
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest_path.exists()
            else {"reservations": {}}
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "guard_state.json"
        self.audit_path = self.state_dir / "risk_audit.jsonl"
        self.interval = max(2, int(os.getenv("GRID_LIVE_GUARD_INTERVAL", "10")))
        self.fail_closed_seconds = max(
            20, int(os.getenv("GRID_LIVE_FAIL_CLOSED_SECONDS", "60"))
        )
        self.technical_refresh_seconds = max(
            30, int(os.getenv("GRID_ROC_BUY_GUARD_REFRESH_SECONDS", "60"))
        )
        self.roc_risk_off_pct = float(os.getenv("GRID_ROC_RISK_OFF_PCT", "-5"))
        self.sqzmom_risk_off_pct = float(os.getenv("GRID_SQZMOM_RISK_OFF_PCT", "-1"))
        self.roc_recovery_pct = float(os.getenv("GRID_ROC_RECOVERY_PCT", "1"))
        self.sqzmom_recovery_pct = float(os.getenv("GRID_SQZMOM_RECOVERY_PCT", "-3"))
        if self.roc_risk_off_pct >= 0 or self.sqzmom_risk_off_pct >= 0:
            raise ValueError("Grid ROC/SQZMOM risk-off thresholds must be negative")
        if self.roc_recovery_pct <= self.roc_risk_off_pct:
            raise ValueError("Grid ROC recovery threshold must exceed the risk-off threshold")
        self.armed = os.getenv("GRID_LIVE_TRADING_ENABLED", "false").lower() == "true"
        self.shadow = os.getenv("GRID_LIVE_GUARD_SHADOW", "false").lower() == "true"
        if self.armed and self.shadow:
            raise ValueError("Grid Guard cannot be armed and shadowed at the same time")
        self.technical_gate_path = self.state_dir / "technical_buy_gate.json"
        self.next_technical_refresh = 0.0
        self.api = ApiClient()
        secret_path = Path(os.getenv(
            "GRID_BINANCE_EMERGENCY_CREDENTIALS_FILE",
            "/run/secrets/grid_binance_emergency_credentials",
        ))
        self.emergency_exchange = (
            BinanceEmergencyClient.from_secret_file(secret_path)
            if secret_path.exists()
            else None
        )
        docker_socket = os.getenv("GRID_DOCKER_SOCKET", "/var/run/docker.sock")
        self.emergency_docker = (
            DockerEmergencyClient(docker_socket) if Path(docker_socket).exists() else None
        )
        manifest_keys = self.manifest.get("reservations", {}).keys()
        self.portfolio_keys = [key for key in manifest_keys if key in PORTFOLIOS]
        if not self.portfolio_keys and self.manifest.get("portfolio") in PORTFOLIOS:
            self.portfolio_keys = [str(self.manifest["portfolio"])]
        if not self.portfolio_keys:
            requested = os.getenv("GRID_LIVE_PORTFOLIOS", "FDUSD")
            self.portfolio_keys = [
                key.strip().upper() for key in requested.split(",")
                if key.strip().upper() in PORTFOLIOS
            ]
        if not self.portfolio_keys:
            raise RuntimeError("Grid Guard has no supported portfolio configured")
        if self.armed:
            if self.emergency_exchange is None or self.emergency_docker is None:
                raise RuntimeError(
                    "armed Grid Guard requires independent Binance credentials "
                    "and the Docker emergency socket"
                )
            pairs = [pair for key in self.portfolio_keys for pair in PORTFOLIOS[key].pairs]
            self.emergency_exchange.verify_ready(pairs)
            for key in self.portfolio_keys:
                self.emergency_docker.matching_containers(PORTFOLIOS[key].bot_name)
        self.state = (
            json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state_path.exists()
            else {"version": 3, "bots": {}, "first_failure_at": None, "last_success_at": 0}
        )
        self.state.update({
            "armed": self.armed,
            "shadow": self.shadow,
            "roc_risk_off_pct": self.roc_risk_off_pct,
            "sqzmom_risk_off_pct": self.sqzmom_risk_off_pct,
            "roc_recovery_pct": self.roc_recovery_pct,
            "sqzmom_recovery_pct": self.sqzmom_recovery_pct,
            "technical_recovery_rule_version": COMBINED_RECOVERY_RULE_VERSION,
        })
        emergency_error = None
        shadow_preflight = None
        if self.shadow and self.emergency_exchange is not None:
            try:
                pairs = [pair for key in self.portfolio_keys for pair in PORTFOLIOS[key].pairs]
                shadow_preflight = self.verify_shadow_exchange_ready(pairs)
            except Exception as exc:
                emergency_error = repr(exc)
        if self.emergency_docker is not None:
            for key in self.portfolio_keys:
                portfolio = PORTFOLIOS[key]
                if self.emergency_docker.matching_containers(portfolio.bot_name):
                    self.state["bots"].setdefault(portfolio.bot_name, {
                        "portfolio": key,
                        "started_at": time.time(),
                        "tripped": False,
                        "action_complete": False,
                    })
            self.state["emergency_ready"] = bool(
                self.emergency_exchange is not None
                and self.emergency_docker is not None
                and emergency_error is None
            )
            self.state["emergency_checked_at"] = time.time()
            if emergency_error:
                self.state["emergency_error"] = emergency_error
            else:
                self.state.pop("emergency_error", None)
            if shadow_preflight is not None:
                self.state["shadow_preflight"] = shadow_preflight
            self.save()

    def save(self):
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def audit(self, event: str, **details):
        with self.audit_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **details,
            }, default=str) + "\n")

    def verify_shadow_exchange_ready(self, pairs: list[str]) -> dict:
        if self.emergency_exchange is None:
            raise RuntimeError("Grid shadow preflight requires Binance credentials")
        self.emergency_exchange.verify_ready(pairs)
        restrictions = self.emergency_exchange._signed(
            "GET", "/sapi/v1/account/apiRestrictions"
        )
        if bool(restrictions.get("enableWithdrawals")):
            raise RuntimeError("Grid Binance key must not permit withdrawals")
        if not bool(restrictions.get("ipRestrict")):
            raise RuntimeError("Grid Binance key must use an IP allowlist")
        if bool(restrictions.get("enableFutures")):
            raise RuntimeError("Grid Binance key must not permit futures trading")
        if bool(restrictions.get("enableMargin")):
            raise RuntimeError("Grid Binance key must not permit margin trading")
        tested = []
        open_order_counts = {}
        commissions = {}
        for pair in pairs:
            open_order_counts[pair] = len(self.emergency_exchange.open_orders(pair))
            fee_payload = self.emergency_exchange._signed(
                "GET", "/sapi/v1/asset/tradeFee",
                {"symbol": pair.replace("-", "")},
            )
            fee_row = fee_payload[0] if isinstance(fee_payload, list) and fee_payload else fee_payload
            commissions[pair] = {
                "maker_fee": str(fee_row["makerCommission"]),
                "taker_fee": str(fee_row["takerCommission"]),
            }
            self.emergency_exchange._signed(
                "POST",
                "/api/v3/order/test",
                {
                    "symbol": pair.replace("-", ""),
                    "side": "BUY",
                    "type": "MARKET",
                    "quoteOrderQty": "10",
                },
            )
            tested.append(pair)
        return {
            "account_read": True,
            "spot_trading": True,
            "withdrawals_disabled": True,
            "ip_restricted": True,
            "futures_disabled": True,
            "margin_disabled": True,
            "test_order_no_fill": True,
            "test_order_pairs": tested,
            "open_order_counts": open_order_counts,
            "commissions": commissions,
        }

    def _technical_gate_targets(self) -> list[Path]:
        targets = [self.technical_gate_path]
        for key in self.portfolio_keys:
            bot_name = PORTFOLIOS[key].bot_name
            for instance in (self.bots_path / "instances").glob(f"{bot_name}*"):
                targets.append(instance / "data" / "technical_buy_gate.json")
        return sorted(set(targets))

    def publish_technical_buy_gate(self, *, force: bool = False) -> dict:
        now = time.time()
        if not force and now < self.next_technical_refresh:
            gate = dict(self.state.get("technical_buy_gate", {}))
            # A new API-created instance can appear between signal refreshes.
            # Republish the last fresh decision every Guard cycle so it starts
            # fail-closed briefly, then receives the gate within one cycle.
            if gate:
                for target in self._technical_gate_targets():
                    atomic_gate_json(target, gate)
            return gate
        self.next_technical_refresh = now + self.technical_refresh_seconds
        previous_gate = self.state.get("technical_buy_gate", {})
        previous_active = bool(previous_gate.get("risk_off_active", False))
        previous_rule_version = previous_gate.get("recovery_rule_version")
        model_changed = previous_rule_version != COMBINED_RECOVERY_RULE_VERSION
        try:
            server_time = requests.get(f"{BINANCE_API}/api/v3/time", timeout=15)
            server_time.raise_for_status()
            server_now_ms = int(server_time.json()["serverTime"])
            response = requests.get(
                f"{BINANCE_API}/api/v3/klines",
                params={"symbol": "BTCFDUSD", "interval": "4h", "limit": 64},
                timeout=15,
            )
            response.raise_for_status()
            closed = [
                item for item in response.json()
                if isinstance(item, list) and len(item) > 6 and int(item[6]) < server_now_ms
            ]
            signal = roc_sqz_signal_from_klines(closed)
            gate = build_technical_buy_gate(
                signal,
                previously_active=previous_active,
                previous_bar_close_time=previous_gate.get(
                    "last_evaluated_bar_close_time"
                ) if not model_changed else None,
                previous_sqzmom_color=previous_gate.get("last_sqzmom_color"),
                roc_risk_off_pct=self.roc_risk_off_pct,
                sqzmom_risk_off_pct=self.sqzmom_risk_off_pct,
                roc_recovery_pct=self.roc_recovery_pct,
                sqzmom_recovery_pct=self.sqzmom_recovery_pct,
            )
        except Exception as exc:
            gate = failed_technical_buy_gate(repr(exc))
        for target in self._technical_gate_targets():
            atomic_gate_json(target, gate)
        old_active = previous_active
        new_active = bool(gate.get("risk_off_active", True))
        self.state["technical_buy_gate"] = gate
        if model_changed or old_active != new_active or not bool(gate.get("source_healthy")):
            self.audit(
                "grid_technical_buy_gate_transition",
                previous_risk_off=old_active,
                risk_off=new_active,
                model_changed=model_changed,
                previous_recovery_rule_version=previous_rule_version,
                gate=gate,
            )
        return gate

    def database(self, bot_name: str) -> Path | None:
        candidates = sorted((self.bots_path / "instances").glob(f"{bot_name}*/data/*.sqlite"),
                            key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    @staticmethod
    def price(pair: str) -> Decimal:
        response = requests.get(f"{BINANCE_API}/api/v3/ticker/price",
                                params={"symbol": pair.replace("-", "")}, timeout=15)
        response.raise_for_status()
        return Decimal(str(response.json()["price"]))

    @staticmethod
    def quantity_step(pair: str) -> Decimal:
        response = requests.get(
            f"{BINANCE_API}/api/v3/exchangeInfo",
            params={"symbol": pair.replace("-", "")},
            timeout=15,
        )
        response.raise_for_status()
        filters = {
            item["filterType"]: item
            for item in response.json()["symbols"][0]["filters"]
        }
        return Decimal(str(filters["LOT_SIZE"]["stepSize"]))

    @staticmethod
    def rows(database: Path, pair: str) -> list[tuple[Any, ...]]:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=10)
        try:
            return connection.execute(
                "SELECT trade_type, price, amount, trade_fee_in_quote FROM TradeFill WHERE symbol=?",
                (pair,),
            ).fetchall()
        finally:
            connection.close()

    def snapshot(self, key: str) -> dict | None:
        portfolio = PORTFOLIOS[key]
        database = self.database(portfolio.bot_name)
        if database is None:
            return None
        pnl, pairs = Decimal("0"), {}
        reservations = self.manifest["reservations"][key]["base"]
        bot_state = self.state.get("bots", {}).get(portfolio.bot_name, {})
        for pair in portfolio.pairs:
            mark = self.price(pair)
            fill_value, net_base = fill_pnl(self.rows(database, pair), mark)
            adjustment_pnl = Decimal("0")
            for adjustment in bot_state.get("emergency_adjustments", []):
                if adjustment.get("pair") != pair:
                    continue
                base_delta = Decimal(str(adjustment["base_delta"]))
                quote_cashflow = Decimal(str(adjustment["quote_cashflow"]))
                fee_quote = Decimal(str(adjustment.get("fee_quote", 0)))
                net_base += base_delta
                adjustment_pnl += quote_cashflow - fee_quote + base_delta * mark
            fill_value += adjustment_pnl
            base = pair.split("-")[0]
            initial_base = Decimal(reservations[base])
            start_price = Decimal(self.manifest["prices"][pair])
            holding_pnl = initial_base * (mark - start_price)
            pair_pnl = fill_value + holding_pnl
            pnl += pair_pnl
            pairs[pair] = {"pnl": str(pair_pnl), "net_base": str(net_base), "mark": str(mark)}
        observed_at = time.time()
        return {
            "pnl": str(pnl),
            "pairs": pairs,
            "database": str(database),
            "updated_at": observed_at,
            "observed_at": observed_at,
            "database_event_at": database.stat().st_mtime,
        }

    @staticmethod
    def _emergency_fill_metrics(pair: str, side: str, response: dict) -> Dict[str, str]:
        executed = Decimal(str(response.get("executedQty", "0")))
        quote = Decimal(str(response.get("cummulativeQuoteQty", "0")))
        if executed <= 0:
            raise RuntimeError(f"emergency order reported no executed quantity: {response}")
        base_asset, quote_asset = pair.split("-", 1)
        base_delta = executed if side == "BUY" else -executed
        quote_cashflow = -quote if side == "BUY" else quote
        fee_quote = Decimal("0")
        for fill in response.get("fills", []):
            commission = Decimal(str(fill.get("commission", "0")))
            asset = str(fill.get("commissionAsset", ""))
            fill_price = Decimal(str(fill.get("price", "0")))
            if asset == quote_asset:
                fee_quote += commission
            elif asset == base_asset:
                base_delta -= commission
                fee_quote += commission * fill_price
        return {
            "base_delta": str(base_delta),
            "quote_cashflow": str(quote_cashflow),
            "fee_quote": str(fee_quote),
        }

    def flatten_deltas(self, key: str, snapshot: dict, bot: dict):
        portfolio = PORTFOLIOS[key]
        results = bot.setdefault("flatten", {})
        for pair, values in snapshot["pairs"].items():
            if pair in results:
                continue
            delta, mark = Decimal(values["net_base"]), Decimal(values["mark"])
            if abs(delta) * mark < Decimal("5.25"):
                results[pair] = "dust"
                self.save()
                continue
            step = self.quantity_step(pair)
            amount = (abs(delta) / step).to_integral_value(rounding=ROUND_DOWN) * step
            if amount <= 0 or amount * mark < Decimal("5.25"):
                results[pair] = "dust_after_exchange_rounding"
                self.save()
                continue
            side = "SELL" if delta > 0 else "BUY"
            if self.emergency_exchange is None:
                raise RuntimeError("independent Binance emergency client is unavailable")
            response = self.emergency_exchange.market_order(pair, side, amount)
            metrics = self._emergency_fill_metrics(pair, side, response)
            adjustment = {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "pair": pair,
                "side": side,
                "order_id": str(response.get("orderId", "")),
                "executed_qty": str(response.get("executedQty", "0")),
                "cummulative_quote_qty": str(response.get("cummulativeQuoteQty", "0")),
                **metrics,
            }
            bot.setdefault("emergency_adjustments", []).append(adjustment)
            results[pair] = {"status": "filled", **adjustment}
            self.save()
        return results

    @staticmethod
    def _items(payload: Any) -> list[dict]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for name in ("data", "items", "containers", "result", "response"):
                nested = payload.get(name)
                values = Guard._items(nested)
                if values:
                    return values
        return []

    def _actual_instances(self, bot_name: str) -> list[str]:
        names = []
        for item in self._items(self.api.active_containers(bot_name)):
            name = str(item.get("name", "")).lstrip("/")
            if name == bot_name or name.startswith(f"{bot_name}-"):
                names.append(name)
        return sorted(set(names))

    def _secure_stop(self, key: str) -> dict:
        if self.emergency_exchange is None or self.emergency_docker is None:
            raise RuntimeError("independent emergency clients are unavailable")
        portfolio = PORTFOLIOS[key]
        instances = set(self.emergency_docker.matching_containers(portfolio.bot_name))
        discovery_errors = []
        try:
            instances.update(self._actual_instances(portfolio.bot_name))
        except Exception as exc:
            discovery_errors.append(repr(exc))
        result = {
            "logical_bot_name": portfolio.bot_name,
            "resolved_instances": sorted(instances),
            "discovery_errors": discovery_errors,
            "graceful_stops": [],
            "exchange_cancellations": {},
            "container_stops": [],
        }
        for candidate in dict.fromkeys([portfolio.bot_name, *sorted(instances)]):
            try:
                result["graceful_stops"].append({
                    "name": candidate,
                    "response": self.api.stop(candidate),
                })
            except Exception as exc:
                result["graceful_stops"].append({"name": candidate, "error": repr(exc)})
        for pair in portfolio.pairs:
            orders = self.emergency_exchange.open_orders(pair)
            result["exchange_cancellations"][pair] = [
                str(item.get("clientOrderId") or item.get("orderId")) for item in orders
            ]
            if orders:
                self.emergency_exchange.cancel_all_orders(pair)
        for instance in sorted(instances):
            result["container_stops"].append({
                "name": instance,
                "response": self.emergency_docker.stop(instance),
            })
        for pair in portfolio.pairs:
            remaining = self.emergency_exchange.open_orders(pair)
            for _ in range(5):
                if not remaining:
                    break
                time.sleep(1)
                self.emergency_exchange.cancel_all_orders(pair)
                remaining = self.emergency_exchange.open_orders(pair)
            if remaining:
                raise RuntimeError(f"active exchange orders remain for {pair}: {remaining}")
        live_instances = self.emergency_docker.matching_containers(portfolio.bot_name)
        for _ in range(5):
            if not live_instances:
                break
            time.sleep(1)
            live_instances = self.emergency_docker.matching_containers(portfolio.bot_name)
        if live_instances:
            raise RuntimeError(f"Grid containers remain live: {live_instances}")
        result["verified_no_active_orders"] = True
        result["verified_no_live_instances"] = True
        return result

    def trip(self, key: str, reason: str, snapshot: Optional[dict]):
        portfolio = PORTFOLIOS[key]
        bot = self.state["bots"].setdefault(portfolio.bot_name, {})
        if getattr(self, "shadow", False):
            observation = {
                "reason": reason,
                "observed_at": time.time(),
                "snapshot": snapshot,
            }
            bot["would_trip"] = observation
            self.audit(
                "grid_circuit_breaker_would_trip",
                bot=portfolio.bot_name,
                **observation,
            )
            self.save()
            return
        if bot.get("action_complete"):
            return
        bot.update({"tripped": True, "reason": reason, "tripped_at": time.time()})
        self.save()
        try:
            stop = self._secure_stop(key)
            # The stopped strategy may have filled or flattened while stopping.
            # Re-read the database before any independent compensation so a
            # stale pre-stop delta can never create the opposite exposure.
            post_stop = None
            for _ in range(5):
                post_stop = self.snapshot(key)
                if post_stop is not None:
                    break
                time.sleep(1)
            if post_stop is None:
                raise RuntimeError("post-stop Grid inventory could not be read")
            bot["flatten"] = {}
            flatten = self.flatten_deltas(key, post_stop, bot)
            reconciled = self.snapshot(key)
            bot.update({
                "stop": stop,
                "stop_complete": True,
                "action_complete": True,
                "post_stop_snapshot": post_stop,
                "reconciled_snapshot": reconciled,
            })
            self.audit("grid_circuit_breaker_complete", bot=portfolio.bot_name,
                       reason=reason, snapshot=snapshot, post_stop=post_stop,
                       reconciled=reconciled, stop=stop, flatten=flatten)
        except Exception as exc:
            bot["action_complete"] = False
            bot["last_action_error"] = repr(exc)
            self.audit("grid_circuit_breaker_action_failed", bot=portfolio.bot_name,
                       reason=reason, error=repr(exc))
        finally:
            self.save()

    def cycle(self):
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        status = json.dumps(self.api.status(), ensure_ascii=True)
        self.publish_technical_buy_gate()
        snapshots = {}
        for key in self.portfolio_keys:
            portfolio = PORTFOLIOS[key]
            docker_running = bool(
                self.emergency_docker
                and self.emergency_docker.matching_containers(portfolio.bot_name)
            )
            if portfolio.bot_name not in status and not docker_running:
                continue
            snapshot = self.snapshot(key)
            if snapshot is None:
                raise RuntimeError(f"Missing SQLite for running bot {portfolio.bot_name}")
            snapshots[key] = snapshot
            bot = self.state["bots"].setdefault(portfolio.bot_name, {})
            budget = budget_for_quote(key)
            equity = budget.capital_limit + Decimal(snapshot["pnl"])
            peak, drawdown = peak_drawdown(
                equity,
                Decimal(str(bot.get("peak_equity", budget.capital_limit))),
                budget.capital_limit,
            )
            snapshot.update({
                "equity": str(equity),
                "peak_equity": str(peak),
                "drawdown_pct": str(drawdown),
            })
            bot["peak_equity"] = str(peak)
            bot["latest"] = snapshot
            pair_limit = budget.pair_loss_limit
            breached_pair = next(
                (
                    pair for pair, values in snapshot["pairs"].items()
                    if Decimal(values["pnl"]) <= -pair_limit
                ),
                None,
            )
            pnl = Decimal(snapshot["pnl"])
            if bot.get("tripped") and not bot.get("action_complete"):
                self.trip(key, bot.get("reason", "retry incomplete breaker"), snapshot)
            elif breached_pair is not None:
                self.trip(
                    key,
                    f"pair PnL {breached_pair} {snapshot['pairs'][breached_pair]['pnl']} "
                    f"<= -{pair_limit} {key}",
                    snapshot,
                )
            elif pnl <= -budget.portfolio_loss_limit:
                self.trip(
                    key,
                    f"portfolio PnL {pnl} <= -{budget.portfolio_loss_limit} {key}",
                    snapshot,
                )
            elif drawdown >= PORTFOLIO_DRAWDOWN_LIMIT_PCT:
                self.trip(
                    key,
                    f"portfolio peak drawdown {drawdown:.2%} >= {PORTFOLIO_DRAWDOWN_LIMIT_PCT:.2%}",
                    snapshot,
                )
        combined_pnl = sum((Decimal(item["pnl"]) for item in snapshots.values()), Decimal("0"))
        self.state["combined_pnl"] = str(combined_pnl)
        if len(snapshots) == len(self.portfolio_keys):
            initial = sum(
                (budget_for_quote(key).capital_limit for key in snapshots), Decimal("0")
            )
            combined_equity = initial + combined_pnl
            combined_peak, combined_drawdown = peak_drawdown(
                combined_equity,
                Decimal(str(self.state.get("combined_peak_equity", initial))),
                initial,
            )
            self.state["combined_peak_equity"] = str(combined_peak)
            self.state["combined_drawdown_pct"] = str(combined_drawdown)
        else:
            combined_drawdown = Decimal("0")
        if (
            len(snapshots) == len(self.portfolio_keys)
            and combined_drawdown >= PORTFOLIO_DRAWDOWN_LIMIT_PCT
        ):
            for key, snapshot in snapshots.items():
                self.trip(
                    key,
                    f"combined peak drawdown {combined_drawdown:.2%} >= "
                    f"{PORTFOLIO_DRAWDOWN_LIMIT_PCT:.2%}",
                    snapshot,
                )
        self.state["first_failure_at"] = None
        self.state["last_success_at"] = time.time()
        self.save()

    def run(self):
        while True:
            try:
                self.cycle()
            except Exception as exc:
                LOG.exception("Grid guard cycle failed")
                self.state["first_failure_at"] = self.state.get("first_failure_at") or time.time()
                self.save()
                if time.time() - self.state["first_failure_at"] >= self.fail_closed_seconds:
                    for key in self.portfolio_keys:
                        portfolio = PORTFOLIOS[key]
                        if portfolio.bot_name in self.state["bots"]:
                            self.trip(
                                key,
                                f"monitor unavailable for {self.fail_closed_seconds} seconds",
                                self.state["bots"][portfolio.bot_name].get("latest"),
                            )
            time.sleep(self.interval)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    armed = os.getenv("GRID_LIVE_TRADING_ENABLED", "false").lower() == "true"
    shadow = os.getenv("GRID_LIVE_GUARD_SHADOW", "false").lower() == "true"
    if not armed and not shadow:
        LOG.warning("Grid live guard is disabled; no API action will be taken.")
        return 0
    Guard().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
