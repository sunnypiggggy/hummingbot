#!/usr/bin/env python3
"""Backtest the disabled BTC/ETH live DCA configuration without deployment code."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


PAIRS = ("BTC-USDT", "ETH-USDT")
STRATEGY_BUDGET_PER_PAIR = 190.0
CAPITAL_LIMIT_PER_PAIR = 200.0


def floor_five_minutes(value: datetime) -> datetime:
    value = value.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return value - timedelta(minutes=value.minute % 5)


def live_config(pair: str) -> dict[str, Any]:
    return {
        "id": f"dca_{pair.lower().replace('-', '')}_live_200_backtest",
        "controller_name": "dman_maker_v2",
        "controller_type": "market_making",
        "connector_name": "binance",
        "trading_pair": pair,
        "total_amount_quote": STRATEGY_BUDGET_PER_PAIR,
        "buy_spreads": [0.0],
        "sell_spreads": [0.0],
        "buy_amounts_pct": [1.0],
        "sell_amounts_pct": [1.0],
        "dca_spreads": [0.01, 0.02, 0.04, 0.08],
        "dca_amounts": [0.10, 0.20, 0.30, 0.40],
        "executor_refresh_time": 300,
        "cooldown_time": 15,
        "leverage": 1,
        "position_mode": "ONEWAY",
        "stop_loss": 0.05,
        "take_profit": 0.02,
        "time_limit": 2700,
        "take_profit_order_type": "LIMIT",
        "skip_rebalance": True,
    }


class Api:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = (os.environ["USERNAME"], os.environ["PASSWORD"])

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        response = self.session.request(
            method, f"{self.base_url}{path}", json=payload, timeout=60
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"{method} {path} failed ({response.status_code}): {response.text[:800]}"
            )
        return response.json()

    def submit(self, config: dict, start: datetime, end: datetime, fee_rate: float) -> str:
        task = self.request("POST", "/backtesting/tasks", {
            "start_time": int(start.timestamp()),
            "end_time": int(end.timestamp()),
            "backtesting_resolution": "5m",
            "trade_cost": fee_rate,
            "config": config,
        })
        return str(task["task_id"])

    def wait(self, task_id: str, timeout: int) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            task = self.request("GET", f"/backtesting/tasks/{task_id}")
            if task.get("status") == "completed":
                return task
            if task.get("status") in {"failed", "cancelled"}:
                raise RuntimeError(
                    f"Backtest {task_id} {task.get('status')}: {task.get('error')}"
                )
            time.sleep(5)
        raise TimeoutError(f"Backtest {task_id} exceeded {timeout} seconds.")


def metrics(task: dict) -> dict:
    result = task.get("result", task)
    result = result.get("results", result)
    return {
        "net_pnl_quote": result.get("net_pnl_quote"),
        "net_pnl_pct_engine": result.get("net_pnl"),
        "max_drawdown_pct": result.get("max_drawdown_pct"),
        "sharpe_ratio": result.get("sharpe_ratio"),
        "total_executors": result.get("total_executors"),
        "executors_with_position": result.get("total_executors_with_position"),
        "close_types": result.get("close_types", {}),
    }


def write_outputs(output: Path, start: datetime, end: datetime, rows: list[dict],
                  raw_tasks: dict[str, dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, task in raw_tasks.items():
        (output / f"{name}_api_result.json").write_text(
            json.dumps(task, indent=2, default=str), encoding="utf-8"
        )
    csv_rows = []
    for row in rows:
        csv_rows.append({
            **row,
            "close_types": json.dumps(row["close_types"], ensure_ascii=True),
        })
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    scenarios = {}
    for fee_label in sorted({row["fee_label"] for row in rows}):
        selected = [row for row in rows if row["fee_label"] == fee_label]
        pnl = sum(float(row["net_pnl_quote"] or 0) for row in selected)
        scenarios[fee_label] = {
            "combined_net_pnl_quote": pnl,
            "return_on_400_capital": pnl / (CAPITAL_LIMIT_PER_PAIR * len(PAIRS)),
            "return_on_380_active_budget": pnl / (STRATEGY_BUDGET_PER_PAIR * len(PAIRS)),
            "worst_pair_drawdown_pct": min(
                float(row["max_drawdown_pct"] or 0) for row in selected
            ),
        }
    run = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "resolution": "5m",
        "deployment_attempted": False,
        "connector_for_market_data": "binance",
        "capital_limit_per_pair": CAPITAL_LIMIT_PER_PAIR,
        "strategy_budget_per_pair": STRATEGY_BUDGET_PER_PAIR,
        "reserve_per_pair": CAPITAL_LIMIT_PER_PAIR - STRATEGY_BUDGET_PER_PAIR,
        "layer_quote_per_side": [9.5, 19.0, 28.5, 38.0],
        "scenarios": scenarios,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")

    headers = list(csv_rows[0])
    table_headers = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    table_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in headers) + "</tr>"
        for row in csv_rows
    )
    scenario_cards = "".join(
        "<article><strong>{}</strong><span>{:+.4f} USDT</span>"
        "<span>{:+.2%} on 400 USDT</span><span>DD {:.2%}</span></article>".format(
            html.escape(name),
            values["combined_net_pnl_quote"],
            values["return_on_400_capital"],
            values["worst_pair_drawdown_pct"],
        )
        for name, values in scenarios.items()
    )
    (output / "report.html").write_text(f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DCA Live Configuration Backtest</title>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:0;background:#f5f7fa;color:#17212b}}
main{{max-width:1200px;margin:auto;padding:24px}}h1{{letter-spacing:0}}.notice{{border-left:5px solid #b54708;background:#fff;padding:14px}}
.cards{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:18px 0}}article{{background:#fff;border:1px solid #d8dee5;border-radius:6px;padding:14px}}
article span{{display:block;margin-top:8px;font-size:18px}}.table{{overflow:auto;background:#fff;border:1px solid #d8dee5}}
table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:8px;border-bottom:1px solid #e5e9ed;white-space:nowrap;text-align:right}}th{{background:#eef2f5}}
code{{background:#e9edf1;padding:2px 5px}}@media(max-width:700px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>DCA 实盘待部署配置回测</h1>
<div class="notice">本报告只调用 Hummingbot API 回测；没有部署 bot，也没有创建真实订单。</div>
<p>{start.isoformat()} 至 {end.isoformat()}，5 分钟 K 线。BTC、ETH 每币总资本 200 USDT，其中 190 USDT 参与策略。</p>
<div class="cards">{scenario_cards}</div>
<p>四层距离：<code>1% / 2% / 4% / 8%</code>；每侧金额：
<code>9.5 / 19 / 28.5 / 38 USDT</code>；止盈 2%，止损 5%，时限 45 分钟，刷新 300 秒。</p>
<div class="table"><table><thead><tr>{table_headers}</tr></thead><tbody>{table_rows}</tbody></table></div>
</main></body></html>""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--fee-rates", default="0.0002,0.001")
    parser.add_argument("--output-root", default="/workspace/results/dca_live_backtests")
    args = parser.parse_args()

    end = floor_five_minutes(datetime.now(timezone.utc))
    start = end - timedelta(days=args.days)
    output = Path(args.output_root) / end.strftime("%Y%m%dT%H%M%SZ")
    api = Api(os.getenv("HUMMINGBOT_API_URL", "http://hummingbot-api:8000"))
    rows: list[dict] = []
    raw_tasks: dict[str, dict] = {}
    for fee_rate in [float(value.strip()) for value in args.fee_rates.split(",") if value.strip()]:
        fee_label = f"fee_{fee_rate:.4%}".replace(".", "_")
        for pair in PAIRS:
            config = live_config(pair)
            task_id = api.submit(config, start, end, fee_rate)
            print(f"submitted {pair} {fee_label}: {task_id}", flush=True)
            task = api.wait(task_id, args.timeout)
            raw_tasks[f"{pair.replace('-', '').lower()}_{fee_label}"] = task
            rows.append({
                "pair": pair,
                "fee_label": fee_label,
                "fee_rate": fee_rate,
                **metrics(task),
            })
            print(f"completed {pair} {fee_label}", flush=True)
    write_outputs(output, start, end, rows, raw_tasks)
    print(json.dumps({"output": str(output), "deployment_attempted": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
