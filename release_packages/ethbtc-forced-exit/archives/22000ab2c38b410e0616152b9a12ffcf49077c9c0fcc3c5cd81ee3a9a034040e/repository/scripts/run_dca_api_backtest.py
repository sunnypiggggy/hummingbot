#!/usr/bin/env python3
"""Run official D-Man Maker V2 API backtests and deploy paper bots after both pass.

This program deliberately does not implement DCA trading logic.  It stages the
repository's official controller in the API-managed controller directory, asks
Hummingbot API to backtest it, and only then asks that same API to deploy two
separate paper-trading controller instances.
"""

import argparse
import csv
import html
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml


LOG = logging.getLogger("dca-api-backtest")
PAIRS = {"BTC-USDT": "dca_btcusdt", "ETH-USDT": "dca_ethusdt"}
FEE_RATE = 0.0002


def five_minute_floor(timestamp: datetime) -> datetime:
    timestamp = timestamp.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return timestamp - timedelta(minutes=timestamp.minute % 5)


def config_for(pair: str, connector: str) -> Dict[str, Any]:
    # The current PaperTradeExchange does not expose live InFlightOrder state
    # to V2 executors. Refreshing a still-open paper DCA executor would create
    # overlapping orders. Backtests retain 300s; paper waits past its 45m TTL.
    refresh_time = 300 if connector == "binance" else int(os.getenv("DCA_PAPER_EXECUTOR_REFRESH_TIME", "3600"))
    return {
        "id": f"dca_{pair.lower().replace('-', '')}",
        "controller_name": "dman_maker_v2",
        "controller_type": "market_making",
        "connector_name": connector,
        "trading_pair": pair,
        "total_amount_quote": 5000,
        "buy_spreads": [0.0],
        "sell_spreads": [0.0],
        "buy_amounts_pct": [1.0],
        "sell_amounts_pct": [1.0],
        "dca_spreads": [0.01, 0.02, 0.04, 0.08],
        "dca_amounts": [0.10, 0.20, 0.30, 0.40],
        "executor_refresh_time": refresh_time,
        "cooldown_time": 15,
        "leverage": 1,
        "position_mode": "ONEWAY",
        "stop_loss": 0.05,
        "take_profit": 0.02,
        "time_limit": 2700,
        "take_profit_order_type": "LIMIT",
        "skip_rebalance": True,
    }


class HummingbotApi:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = (os.environ["USERNAME"], os.environ["PASSWORD"])

    def request(self, method: str, path: str, payload: Dict[str, Any] = None) -> Dict[str, Any]:
        response = self.session.request(method, f"{self.base_url}{path}", json=payload, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(f"{method} {path} failed ({response.status_code}): {response.text[:1000]}")
        return response.json()

    def backtest(self, config: Dict[str, Any], start: datetime, end: datetime) -> str:
        task = self.request("POST", "/backtesting/tasks", {
            "start_time": int(start.timestamp()),
            "end_time": int(end.timestamp()),
            "backtesting_resolution": "5m",
            "trade_cost": FEE_RATE,
            "config": config,
        })
        return task["task_id"]

    def await_task(self, task_id: str, timeout_seconds: int) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            task = self.request("GET", f"/backtesting/tasks/{task_id}")
            if task.get("status") == "completed":
                return task
            if task.get("status") in {"failed", "cancelled"}:
                raise RuntimeError(f"Backtest task {task_id} {task.get('status')}: {task.get('error')}")
            time.sleep(5)
        raise TimeoutError(f"Backtest task {task_id} did not finish within {timeout_seconds}s")

    def deploy(self, instance_name: str, controller_config: str) -> Dict[str, Any]:
        return self.request("POST", "/bot-orchestration/deploy-v2-controllers", {
            "instance_name": instance_name,
            "credentials_profile": "master_account",
            "controllers_config": [controller_config],
            "image": os.getenv("DCA_RUNTIME_IMAGE", "hummingbot/dca-paper-runtime:local"),
            "headless": True,
        })


def stage_controller_and_configs(bots: Path) -> None:
    controller_dir = bots / "controllers" / "market_making"
    config_dir = bots / "conf" / "controllers"
    scripts_dir = bots / "scripts"
    profile_dir = bots / "credentials" / "master_account"
    for directory in (controller_dir, config_dir, scripts_dir, profile_dir):
        directory.mkdir(parents=True, exist_ok=True)
    Path(controller_dir.parent / "__init__.py").touch()
    Path(controller_dir / "__init__.py").touch()
    shutil.copy2("/app/dman_maker_v2.py", controller_dir / "dman_maker_v2.py")
    # API mounts this shared scripts directory into every runtime container.
    # Stage the stock V2 host script here so the mount does not hide the copy
    # that is bundled in hummingbot/hummingbot:latest.
    shutil.copy2("/app/v2_with_controllers.py", scripts_dir / "v2_with_controllers.py")
    for pair, filename in PAIRS.items():
        config = config_for(pair, "binance_paper_trade")
        (config_dir / f"{filename}.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    # The credential profile is paper-only and must never contain live Binance keys.
    if not (profile_dir / "conf_client.yml").exists():
        paper_balances = {"USDT": 100000, "BTC": 1, "ETH": 20}
        (profile_dir / "conf_client.yml").write_text(yaml.safe_dump({
            "instance_id": "api-managed-dca", "log_level": "INFO", "db_mode": {"db_engine": "sqlite"},
            "paper_trade": {"paper_trade_exchanges": ["binance"], "paper_trade_account_balance": paper_balances},
            "mqtt_bridge": {"mqtt_host": "127.0.0.1", "mqtt_port": 1883, "mqtt_namespace": "hbot",
                            "mqtt_ssl": False, "mqtt_autostart": False},
        }, sort_keys=False), encoding="utf-8")
    (profile_dir / "conf_fee_overrides.yml").write_text(
        "template_version: 14\nbinance_maker_percent_fee: 0.02\nbinance_taker_percent_fee: 0.02\n",
        encoding="utf-8",
    )


def result_metrics(task: Dict[str, Any]) -> Dict[str, Any]:
    result = task.get("result", task)
    results = result.get("results", result)
    return {
        "net_pnl_quote": results.get("net_pnl_quote"), "net_pnl": results.get("net_pnl"),
        "max_drawdown_pct": results.get("max_drawdown_pct"), "sharpe_ratio": results.get("sharpe_ratio"),
        "total_executors": results.get("total_executors"),
        "total_executors_with_position": results.get("total_executors_with_position"),
        "close_types": results.get("close_types", {}),
    }


def write_report(output: Path, started: datetime, ended: datetime, tasks: Dict[str, Dict[str, Any]], deployments: Dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    for pair, task in tasks.items():
        (output / f"{pair.replace('-', '').lower()}_api_result.json").write_text(json.dumps(task, indent=2, default=str), encoding="utf-8")
        row = {"pair": pair, **result_metrics(task)}
        row["close_types"] = json.dumps(row["close_types"], ensure_ascii=False)
        summary_rows.append(row)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    total_quote = sum(float(row["net_pnl_quote"] or 0) for row in summary_rows)
    total_pct = total_quote / 10000
    max_drawdown = min(float(row["max_drawdown_pct"] or 0) for row in summary_rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "start": started.isoformat(), "end": ended.isoformat(),
        "fee_rate": FEE_RATE, "allocation": {"BTC-USDT": 5000, "ETH-USDT": 5000},
        "combined_net_pnl_quote": total_quote, "combined_net_pnl_pct": total_pct,
        "conservative_max_drawdown_pct": max_drawdown, "deployments": deployments,
    }
    (output / "run.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in row) + "</tr>" for row in summary_rows)
    headers = "".join(f"<th>{html.escape(key)}</th>" for key in summary_rows[0])
    (output / "report.html").write_text(f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<title>BTC/ETH DCA API Backtest</title><style>body{{font-family:Arial,sans-serif;margin:32px;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #d8dee9;text-align:left}}th{{background:#eef3f8}}.metric{{font-size:20px;font-weight:700}}</style>
<h1>官方 API D-Man Maker V2 现货 DCA 回测</h1><p>{started.isoformat()} 至 {ended.isoformat()}，5 分钟 K 线，手续费 0.02%。</p>
<p class='metric'>组合净收益：{total_quote:.4f} USDT ({total_pct:.2%})</p><p>保守最大回撤（两币较大值）：{max_drawdown:.2%}</p>
<p>参数：每币 5000 USDT；DCA 1/2/4/8%；资金 10/20/30/40%；止盈 2%；止损 5%；45 分钟时限；300 秒刷新。</p>
<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>
<p>部署状态：{html.escape(json.dumps(deployments, ensure_ascii=False))}</p></html>""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--deploy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", default="/workspace/results/dca_api_backtests")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    end = five_minute_floor(datetime.now(timezone.utc))
    start = end - timedelta(days=args.days)
    output = Path(args.output_root) / end.strftime("%Y%m%dT%H%M%SZ")
    bots = Path(os.getenv("BOTS_PATH", "/workspace/bots"))
    stage_controller_and_configs(bots)
    api = HummingbotApi(os.getenv("HUMMINGBOT_API_URL", "http://hummingbot-api:8000"))
    tasks: Dict[str, Dict[str, Any]] = {}
    submitted: Dict[str, str] = {}
    try:
        # The official API owns one stateful backtesting engine. Submit and
        # finish one task at a time so BTC and ETH cannot overwrite its candle
        # window or controller state.
        for pair in PAIRS:
            submitted[pair] = api.backtest(config_for(pair, "binance"), start, end)
            LOG.info("Submitted %s backtest task %s", pair, submitted[pair])
            tasks[pair] = api.await_task(submitted[pair], args.timeout)
            LOG.info("Completed %s backtest task %s", pair, submitted[pair])
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        (output / "failure.json").write_text(json.dumps({"error": str(exc), "submitted": submitted}, indent=2), encoding="utf-8")
        LOG.exception("DCA backtest failed. No paper bot will be deployed.")
        return 1
    deployments: Dict[str, Any] = {"status": "not_requested"}
    if args.deploy:
        deployments = {}
        for pair, filename in PAIRS.items():
            deployments[pair] = api.deploy(f"dca-{pair.lower().replace('-', '')}", f"{filename}.yml")
            LOG.info("Deployed paper instance for %s", pair)
    write_report(output, start, end, tasks, deployments)
    LOG.info("DCA API backtest report written to %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
