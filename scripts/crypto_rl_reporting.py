"""Auditable hourly-accounting metrics for the isolated crypto RL study.

Returns are fractions. Equity rows are labelled by hour open and valued at hour
close. No round-trip win rate is invented for a portfolio that scales positions.
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
    "ppo": "PPO单种子", "ppo_ensemble": "PPO五模型平均权重组合", "cash": "USDT现金", "top10_equal": "动态Top10等权（限仓版）",
    "top10_cap": "动态Top10市值加权（限仓版）", "btc_eth_hold": "BTC／ETH等权持有（被动参考，无25%单币上限）",
    "equal_weight": "动态Top10等权", "cap_weight": "动态Top10市值加权",
    "buy_hold_equal": "BTC／ETH等权持有",
}


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def canonical_csv(frame: pd.DataFrame, *, index=False) -> bytes:
    return frame.to_csv(index=index, index_label="timestamp" if index else None,
                        float_format="%.17g", lineterminator="\n").encode("utf-8")


def result_fingerprints(result: dict) -> dict:
    return {name: hashlib.sha256(canonical_csv(result[name], index=name in ("equity", "boundary_equity"))).hexdigest()
            for name in ("equity", "fills", "weights", "boundary_equity") if isinstance(result.get(name), pd.DataFrame)}


def save_result(path: Path, result: dict) -> dict:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    fingerprints = result_fingerprints(result)
    with (path / "equity_1h.csv.gz").open("wb") as stream:
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as zipped:
            zipped.write(canonical_csv(result["equity"], index=True))
    for name in ("fills", "weights"):
        frame = result.get(name)
        if isinstance(frame, pd.DataFrame):
            (path / f"{name}.csv").write_bytes(canonical_csv(frame))
        elif frame is not None:
            write_json(path / f"{name}.json", frame)
    if isinstance(result.get("boundary_equity"), pd.DataFrame):
        (path / "boundary_equity.csv").write_bytes(canonical_csv(result["boundary_equity"], index=True))
    write_json(path / "metadata.json", result.get("metadata", {}))
    write_json(path / "fingerprints.json", fingerprints)
    return fingerprints


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def summarize(result: dict, *, run_id: str, strategy: str, seed=None, scenario: str,
              split: str, start, end, capital=10000., fee=.001,
              slippage=.0002) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Return exact summary, UTC daily equity and monthly return records.

    A complete hourly window is mandatory. Fees/notional come from fills rather
    than training reward. Net liquidation is a separately labelled hypothetical
    quote-paid exit, preserving the actual holdings and marked equity.
    """
    start, end = _utc(start), _utc(end)
    if start >= end or start != start.floor("h") or end != end.floor("h") or capital <= 0:
        raise ValueError("Metrics require positive capital and an hourly [start,end) window")
    if not 0 <= fee < 1 or not 0 <= slippage < 1:
        raise ValueError("Invalid fractional cost assumption")
    frame = result["equity"]
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("Hourly equity must have timezone-aware opening timestamps")
    expected = pd.date_range(start, end, freq="h", inclusive="left")
    if not frame.index.tz_convert("UTC").equals(expected):
        raise ValueError("Hourly equity must cover every hour exactly once")
    missing = {"equity", "cash", "base_value", "exposure"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing equity columns: {sorted(missing)}")
    values = frame[["equity", "cash", "base_value", "exposure"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (frame.equity <= 0).any() or (frame.cash < -1e-7).any() or (frame.base_value < -1e-7).any():
        raise AssertionError("Non-finite or negative spot account state")
    tolerance = max(1e-6, capital * 1e-9)
    if (frame.equity - frame.cash - frame.base_value).abs().max() > tolerance:
        raise AssertionError("Cash plus marked holdings does not reconcile to equity")
    if (frame.exposure - frame.base_value / frame.equity).abs().max() > 1e-8:
        raise AssertionError("Reported exposure does not reconcile")
    metadata = result.get("metadata", {})
    initial = float(metadata.get("initial_capital", capital))
    if abs(initial - capital) > tolerance:
        raise AssertionError("Run capital differs from metric comparison capital")
    fills = result.get("fills", pd.DataFrame())
    if len(fills):
        if not {"qty", "price", "fee", "side", "timestamp"}.issubset(fills.columns):
            raise ValueError("Fills must expose timestamp/side/qty/price/fee")
        numbers = fills[["qty", "price", "fee"]].to_numpy(dtype=float)
        if not np.isfinite(numbers).all() or (fills.qty <= 0).any() or (fills.price <= 0).any() or (fills.fee < 0).any():
            raise AssertionError("Invalid fill economics")
        times = pd.to_datetime(fills.timestamp, utc=True)
        if not ((times >= start) & (times < end)).all():
            raise AssertionError("Fill outside the evaluation window")
        if not fills.side.isin(["BUY", "SELL"]).all():
            raise AssertionError("Unknown fill side")
        fees = float(fills.fee.sum())
        notional = float((fills.qty * fills.price).sum())
        if abs(fees - notional * fee) > tolerance:
            raise AssertionError("Fill fees differ from the declared per-side research rate")
    else:
        fees, notional = 0., 0.
    for key, calculated in (("fees", fees), ("total_notional", notional)):
        if key in metadata and abs(float(metadata[key]) - calculated) > tolerance:
            raise AssertionError(f"Serialized {key} does not reconcile to fills")
    if len(fills) and "slippage_cost" in fills.columns and metadata.get("slippage_cost") is not None:
        if abs(float(fills.slippage_cost.sum()) - float(metadata["slippage_cost"])) > tolerance:
            raise AssertionError("Serialized slippage cost does not reconcile to fills")
    equity = frame.equity.astype(float)
    final = float(equity.iloc[-1])
    final_cash, final_base = float(frame.cash.iloc[-1]), float(frame.base_value.iloc[-1])
    liquidation = final_cash + final_base * (1 - slippage) * (1 - fee)
    if "final_liquidation_equity" in metadata and abs(float(metadata["final_liquidation_equity"]) - liquidation) > tolerance:
        raise AssertionError("End liquidation differs from the common exit-cost convention")
    peaks = equity.cummax().clip(lower=initial)
    drawdown = equity / peaks - 1
    daily = equity.resample("1D").last()
    daily_returns = daily.pct_change()
    daily_returns.iloc[0] = daily.iloc[0] / initial - 1
    deviation = float(daily_returns.std(ddof=1))
    sharpe = float(daily_returns.mean() / deviation * np.sqrt(365)) if deviation > 1e-15 else None
    days = (end - start).total_seconds() / 86400
    log_annual_return = math.log(final / initial) * 365 / days
    annual_return = math.expm1(log_annual_return) if log_annual_return < math.log(np.finfo(float).max) else None
    max_drawdown = float(drawdown.min())
    label = LABELS.get(strategy, strategy)
    if seed is not None and not pd.isna(seed):
        label = f"{label} · seed {int(seed)}"
    row = {
        "run_id": run_id, "strategy": strategy, "label": label,
        "seed": int(seed) if seed is not None and not pd.isna(seed) else None,
        "scenario": scenario, "split": split, "initial_capital": initial,
        "start": start.isoformat(), "end_exclusive": end.isoformat(),
        "final_equity": final, "net_pnl": final - initial, "net_return": final / initial - 1,
        "max_drawdown": max_drawdown, "drawdown_grain": "1h_close",
        "sharpe_daily": sharpe, "cagr": annual_return,
        "calmar": annual_return / abs(max_drawdown) if annual_return is not None and max_drawdown < -1e-15 else None,
        "fills": len(fills), "buy_fills": int(fills.side.eq("BUY").sum()) if len(fills) else 0,
        "sell_fills": int(fills.side.eq("SELL").sum()) if len(fills) else 0,
        "fees": fees, "turnover": notional / initial, "total_notional": notional,
        "average_exposure": float(frame.exposure.mean()), "exposure": float(frame.exposure.mean()),
        "final_cash": final_cash, "final_base_value": final_base,
        "net_liquidation_equity": liquidation, "net_liquidation_return": liquidation / initial - 1,
        "slippage_cost": metadata.get("slippage_cost"), "observed_hours": len(frame),
    }
    daily_drawdown = drawdown.resample("1D").last()
    common = {key: row[key] for key in ("run_id", "strategy", "label", "seed", "scenario", "split")}
    history = pd.DataFrame({
        "timestamp": pd.DatetimeIndex([min(day + pd.Timedelta(days=1), end) for day in daily.index]), "equity": daily.to_numpy(),
        "net_return": daily.to_numpy() / initial - 1, "drawdown": daily_drawdown.to_numpy(), **common,
    })
    initial_row = {"timestamp": start, "equity": initial, "net_return": 0., "drawdown": 0., **common}
    history = pd.concat([pd.DataFrame([initial_row]), history], ignore_index=True)
    monthly_values = daily.resample("ME").last()
    monthly_returns = monthly_values.pct_change()
    monthly_returns.iloc[0] = monthly_values.iloc[0] / initial - 1
    monthly = pd.DataFrame({"month": monthly_values.index.strftime("%Y-%m"), "return": monthly_returns.to_numpy(), **common})
    return clean_json(row), history, monthly


def seed_statistics(summary: pd.DataFrame) -> pd.DataFrame:
    """Descriptive dispersion across all PPO seeds; never select a test winner."""
    rows = []
    selected = summary.loc[summary.seed.notna()]
    for (split, scenario), group in selected.groupby(["split", "scenario"], sort=True):
        rows.append({
            "split": split, "scenario": scenario, "seeds": len(group),
            "median_net_return": float(group.net_return.median()), "mean_net_return": float(group.net_return.mean()),
            "worst_net_return": float(group.net_return.min()), "best_net_return": float(group.net_return.max()),
            "median_max_drawdown": float(group.max_drawdown.median()), "worst_max_drawdown": float(group.max_drawdown.min()),
            "positive_seeds": int(group.net_return.gt(0).sum()),
        })
    return pd.DataFrame(rows)


def make_reviewed_snapshot(summary: pd.DataFrame, history: pd.DataFrame, monthly: pd.DataFrame,
                           universe: pd.DataFrame, training: pd.DataFrame, *, manifest: dict,
                           holdings: pd.DataFrame | None = None) -> dict:
    """Create the report's five source-scoped queries from reviewed local outputs."""
    config = manifest["config"]
    start = config.get("test_start", config.get("start", "2025-10-23T00:00:00Z"))
    end = config.get("test_end", config.get("end", "2026-10-23T00:00:00Z"))
    sources = {
        "label": "历史动态Top10与本地PPO现货组合模拟",
        "files": ["summary.csv", "daily_equity.csv", "monthly_returns.csv", "manifest.json"],
        "period": {"start": start, "end": end}, "timezone": "UTC", "classification": "modeled",
        "evidenceFlow": [
            {"title": "历史资产池", "detail": "按已保存CMC周快照日期加2天滞后，月初及切分起点选择；剔除稳定币、包装资产，并要求当时Binance现货可用。快照为后取得的历史回溯，不能证明原始发布时间point-in-time。"},
            {"title": "成交与估值", "detail": "1小时行情，日频动作；00:00卖出，下一小时01:00以净现金买入，逐腿扣手续费与不利滑点；完整小时收盘权益计算回撤。"},
            {"title": "学习与评估", "detail": "训练、90天验证和固定365天测试分离；5个随机种子全部保留，压力成本复用冻结模型，测试不参与选种子或调参。"},
        ],
        "caveats": ["市值资产池是历史回溯版本，不能证明原始发布时即可获得的严格point-in-time数据。", "回撤基于1小时收盘权益，不能与前轮分钟最大回撤视为相同采样口径。", "费用与滑点是研究假设；没有历史盘口、队列及全部历史过滤器重建。", "种子分布描述同一市场历史上的训练波动，不是五份独立市场样本。", "模型训练不读取测试数据，但研究者已看过本年其他策略表现，不能宣称完全独立的新研究样本。", "BTC/ETH各47.5%的被动持有参考不遵守策略25%单币目标上限；其费用和执行时序保持一致。"],
        "metricDefinitions": [
            {"label": "净收益", "definition": "期末现金与持币市值相对初始资本变化，已扣模拟成交费用及不利滑点。"},
            {"label": "最大回撤", "definition": "完整1小时收盘权益相对历史峰值的最大跌幅，历史峰值包含期初本金。"},
            {"label": "日收益Sharpe", "definition": "UTC日末权益收益均值/样本标准差×sqrt(365)，包含首日成本，无风险收益设0。"},
            {"label": "末端净清算", "definition": "保留实际期末持仓估值，另按统一退出手续费和滑点估算净清算权益；未执行真实清仓。"},
        ],
    }
    records = lambda frame: json.loads(frame.to_json(orient="records", date_format="iso"))
    queries = {}
    for query_id, frame, files in (
        ("strategy_summary", summary, ["summary.csv"]),
        ("equity_history", history, ["daily_equity.csv"]),
        ("monthly_returns", monthly, ["monthly_returns.csv"]),
        ("universe_history", universe, ["universe_history.csv"]),
        ("training_metrics", training, ["training_metrics.csv"]),
    ):
        source = json.loads(json.dumps(sources, ensure_ascii=False))
        source["files"] = [*files, "manifest.json"]
        if query_id in ("strategy_summary", "monthly_returns") and len(summary) and {"start", "end_exclusive"}.issubset(summary.columns):
            source["period"] = {"start": min(summary.start), "end": max(summary.end_exclusive)}
        elif query_id == "equity_history" and len(history):
            source["period"] = {"start": str(pd.to_datetime(history.timestamp, utc=True).min()),
                                "end": str(pd.to_datetime(history.timestamp, utc=True).max())}
        elif query_id == "training_metrics":
            source["period"] = {"start": "2023-01-01T00:00:00Z", "end": "2025-07-25T00:00:00Z"}
        elif query_id == "universe_history" and len(universe):
            date_field = next((field for field in ("anchor", "date", "timestamp") if field in universe.columns), None)
            if date_field:
                source["period"] = {"start": str(pd.to_datetime(universe[date_field], utc=True).min()), "end": str(end)}
        queries[query_id] = {"rows": records(frame), "source": source}
    if holdings is not None:
        source = json.loads(json.dumps(sources, ensure_ascii=False))
        source["files"] = ["final_holdings.csv", "manifest.json"]
        queries["final_holdings"] = {"rows": records(holdings), "source": source}
    return {"surface": "report", "title": "动态Top10：PPO组合的扣费收益与回撤", "buildStatus": "complete",
            "generatedAt": manifest.get("created_at"), "status": "reviewed",
            "report": {"asOf": (_utc(end) - pd.Timedelta(hours=1)).strftime("%Y-%m-%d"), "title": "动态Top10：PPO组合的扣费收益与回撤"},
            "filters": [], "queries": queries}
