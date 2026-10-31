"""Reporting and reconciliation for the isolated spot strategy experiment.

This module performs no trading, private API calls, or production-state reads.
Returns stored in summary tables are fractions; trades.return_pct is percent.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LABELS = {
    "donchian": "Donchian 趋势突破", "cluc": "Cluc 均值回归",
    "rsi2": "RSI2 趋势回调", "macd_ema": "MACD＋EMA 动量",
    "rotation": "BTC／ETH／USDT 轮动", "buy_hold": "买入持有",
    "buy_hold_equal": "BTC／ETH 等权持有", "cash": "现金",
}


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def canonical_csv(frame: pd.DataFrame, *, index: bool = False) -> bytes:
    return frame.to_csv(index=index, index_label="timestamp" if index else None,
                        float_format="%.12g", lineterminator="\n").encode("utf-8")


def result_fingerprints(result: dict) -> dict:
    return {key: hashlib.sha256(canonical_csv(result[key], index=key == "equity")).hexdigest()
            for key in ("equity", "fills", "trades")}


def combine_sleeves(results: list[dict]) -> dict:
    first = results[0]["equity"].index
    if any(not r["equity"].index.equals(first) for r in results):
        raise ValueError("Cannot combine sleeves with different timestamps")
    equity = sum((r["equity"][["equity", "cash", "base_value"]] for r in results))
    equity["exposure"] = equity["base_value"] / equity["equity"]
    metadata = {key: sum(float(r["metadata"].get(key, 0)) for r in results)
                for key in ("initial_capital", "final_cash", "fees", "total_notional", "slippage_cost",
                            "final_liquidation_equity", "unrealized_pnl")}
    metadata["final_positions"] = {k: v for r in results for k, v in r["metadata"]["final_positions"].items()}
    metadata["equity_timestamp_semantics"] = "minute open label; value marked at that minute close"
    return {"equity": equity,
            "fills": pd.concat([r["fills"] for r in results], ignore_index=True).sort_values("timestamp", kind="stable").reset_index(drop=True),
            "trades": pd.concat([r["trades"] for r in results], ignore_index=True).sort_values("entry_time", kind="stable").reset_index(drop=True),
            "metadata": metadata}


def summarize(result: dict, *, run_id: str, strategy: str, pair: str, scenario: str,
              variant: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    equity = result["equity"]["equity"]
    meta, trades = result["metadata"], result["trades"]
    initial = float(meta["initial_capital"])
    final = float(equity.iloc[-1])
    drawdown = equity / equity.cummax().clip(lower=initial) - 1
    max_drawdown = float(drawdown.min())
    daily = equity.resample("1D").last()
    daily_return = daily.pct_change()
    daily_return.iloc[0] = daily.iloc[0] / initial - 1
    sd = float(daily_return.std(ddof=1))
    sharpe = float(daily_return.mean() / sd * np.sqrt(365)) if sd > 1e-15 else None
    days = (end - start).total_seconds() / 86400
    annual_log_return = (math.log(final) - math.log(initial)) * 365 / days if final > 0 else -math.inf
    cagr = math.expm1(annual_log_return) if annual_log_return < math.log(np.finfo(float).max) else None
    pnl = trades["pnl"].astype(float) if len(trades) else pd.Series(dtype=float)
    gains, losses = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
    mean_gain = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else None
    mean_loss = float(-pnl[pnl < 0].mean()) if (pnl < 0).any() else None
    realized = float(pnl.sum())
    unrealized = float(meta.get("unrealized_pnl", 0))
    bridge_error = final - initial - realized - unrealized
    if abs(bridge_error) > max(1e-6, initial * 1e-9):
        raise AssertionError(f"{run_id}: realized/unrealized PnL does not reconcile: {bridge_error}")
    row = {
        "run_id": run_id, "strategy": strategy, "label": LABELS[strategy], "pair": pair,
        "scenario": scenario, "variant": variant, "initial_capital": initial,
        "final_equity": final, "net_pnl": final - initial, "net_return": final / initial - 1,
        "cagr": cagr, "max_drawdown": max_drawdown, "sharpe_daily": sharpe,
        "calmar": cagr / abs(max_drawdown) if cagr is not None and max_drawdown < -1e-15 else None,
        "closed_trades": len(trades), "win_rate": float((pnl > 0).mean()) if len(pnl) else None,
        "profit_factor": gains / losses if losses > 0 else None,
        "payoff_ratio": mean_gain / mean_loss if mean_gain is not None and mean_loss else None,
        "fees": float(meta["fees"]), "slippage_cost": float(meta["slippage_cost"]),
        "turnover": float(meta["total_notional"]) / initial,
        "average_holding_hours": float(trades.holding_minutes.mean() / 60) if len(trades) else None,
        "exposure": float(result["equity"].exposure.mean()), "realized_pnl": realized,
        "unrealized_pnl": unrealized, "final_cash": float(meta["final_cash"]),
        "net_liquidation_equity": float(meta["final_liquidation_equity"]),
        "net_liquidation_return": float(meta["final_liquidation_equity"]) / initial - 1,
        "reconciliation_error": bridge_error,
    }
    # Include the initial capital as a point before the first close-to-close return.
    history = pd.DataFrame({"timestamp": [start.isoformat()] + [(t + pd.Timedelta(days=1)).isoformat() for t in daily.index],
                            "equity": [initial] + daily.tolist(),
                            "net_return": [0.0] + (daily / initial - 1).tolist(),
                            "drawdown": [0.0] + drawdown.resample("1D").last().tolist()})
    for key in ("run_id", "strategy", "pair", "scenario", "variant", "label"):
        history[key] = row[key]
    month_end = equity.resample("ME").last()
    prev = np.r_[initial, month_end.values[:-1]]
    monthly = pd.DataFrame({"month": month_end.index.strftime("%Y-%m"), "return": month_end.values / prev - 1,
                            "pnl": month_end.values - prev, "end_equity": month_end.values})
    for key in ("run_id", "strategy", "pair", "scenario", "variant", "label"):
        monthly[key] = row[key]
    return clean_json(row), history, monthly


def save_result(path: Path, result: dict) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    fingerprints = result_fingerprints(result)
    for key in ("fills", "trades"):
        (path / f"{key}.csv").write_bytes(canonical_csv(result[key]))
    with (path / "equity_1m.csv.gz").open("wb") as stream:
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as zipped:
            zipped.write(canonical_csv(result["equity"], index=True))
    write_json(path / "metadata.json", result["metadata"])
    write_json(path / "fingerprints.json", fingerprints)
    return fingerprints


def make_reviewed_snapshot(summary: pd.DataFrame, history: pd.DataFrame, monthly: pd.DataFrame,
                           *, manifest: dict) -> dict:
    source = {
        "label": "Binance 现货 K 线与固定规则回测",
        "files": ["summary.csv", "daily_equity.csv", "monthly_returns.csv", "manifest.json"],
        "links": [{"label": "Binance 官方公开数据", "url": "https://github.com/binance/binance-public-data"}],
        "period": {"start": manifest["config"]["start"], "end": manifest["config"]["end"]},
        "timezone": "UTC", "classification": "modeled",
        "evidenceFlow": [
            {"title": "行情", "detail": "现货1分钟OHLCV；原始ZIP校验SHA256，缺口失败，不插值；另取210天预热。"},
            {"title": "执行", "detail": "仅已闭合信号，下一分钟open加不利滑点，双边taker手续费；现货现金/持币非负。"},
            {"title": "比较", "detail": "每组10,000 USDT；前4组各币5,000独立资金，第5组轮动共享资金；原退出与5%保护版分别报告。"},
        ],
        "caveats": ["费用与滑点为情景假设，不是账户全年实际成本。", "仅此365天回测，不证明未来收益或实盘可执行性。",
                    "图表为日末权益；最大回撤使用1分钟权益，含期初资金与浮盈亏。", "各组同为现货多头，收益并非独立。"],
        "metricDefinitions": [
            {"label": "净收益率", "definition": "期末现金加持币市值相对初始资金的变化，已扣实际模拟成交费用与滑点。"},
            {"label": "最大回撤", "definition": "1分钟收盘权益相对历史峰值（包含期初资金）的最大跌幅。"},
            {"label": "Sharpe", "definition": "UTC日末权益收益（包含首日成本），均值/样本标准差×sqrt(365)，无风险利率设0。"},
            {"label": "Calmar", "definition": "年化收益率/最大回撤绝对值；零回撤时缺失。"},
        ],
    }
    records = lambda df: json.loads(df.to_json(orient="records", date_format="iso"))
    return {"surface": "report", "title": "五组现货策略的365天比较", "buildStatus": "complete",
            "generatedAt": manifest["created_at"], "status": "reviewed",
            "report": {"asOf": (pd.Timestamp(manifest["config"]["end"]) - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d"),
                       "title": "五组现货策略的365天比较"},
            "filters": [], "queries": {
                "strategy_summary": {"rows": records(summary), "source": source},
                "equity_history": {"rows": records(history), "source": source},
                "monthly_returns": {"rows": records(monthly), "source": source},
            }}


def write_readout(path: Path, summary: pd.DataFrame, config: dict) -> None:
    main = summary[(summary.pair == "PORTFOLIO") & (summary.scenario == "base") & (summary.variant == "original")]
    main = main[~main.strategy.isin(["cash", "buy_hold_equal"])].sort_values("net_return", ascending=False)
    lines = ["# 五组现货策略：365天回测", "", f"评估区间：{config['start']} 至 {config['end']}（右端不含），UTC。", "",
             "## 主成本／原退出规则", "", "| 策略 | 净收益 | 最大回撤 | Sharpe | 交易数 | 费用USDT |", "|---|---:|---:|---:|---:|---:|"]
    for row in main.to_dict("records"):
        sh = f"{row['sharpe_daily']:.2f}" if pd.notna(row["sharpe_daily"]) else "—"
        lines.append(f"| {row['label']} | {row['net_return']:.2%} | {row['max_drawdown']:.2%} | {sh} | {row['closed_trades']} | {row['fees']:.2f} |")
    base, stress = config["costs"]["base"], config["costs"]["stress"]
    lines += ["", "完整对照见 summary.csv；逐笔和分钟权益见 runs/。", "",
              f"- 主成本单边手续费{base['fee']:.2%}、滑点{base['slippage']:.2%}；压力成本{stress['fee']:.2%}、{stress['slippage']:.2%}，均为研究假设。",
              f"- 每组{config['capital']:,.0f} USDT，前四组各币{config['sleeve_capital']:,.0f} USDT；空仓时最多投入可用现金95%，不加仓、不借币。",
              "- 原退出保留Cluc的净ROI1%/5%止损；其余主版无额外价格止损。保护版统一加入5%止损。",
              "- 期末未强平，净收益包含浮盈亏；net_liquidation_return另计末端卖出成本。",
              "- 胜率、盈亏比及持仓时间只统计已平仓交易；总收益包括未平仓交易。",
              "- 历史手续费/盘口未获完整重建；当前交易过滤器仅为带采集时间的快照。",
              "- 各源策略均有明确改版/执行归一化，不能继承源项目宣传收益；未授权实盘。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
