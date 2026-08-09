"""Mobile PNG/PDF renderer for hash-bound v22 parameter evidence."""

from __future__ import annotations

import csv
import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests
from PIL import Image, ImageDraw

try:
    from telegram_notifications import (
        report_font, render_analysis_pdf, sha256_file,
    )
except ModuleNotFoundError:
    from live_guard.telegram_notifications import (
        report_font, render_analysis_pdf, sha256_file,
    )


WIDTH, HEIGHT = 1440, 2400
DAY = 86400
WINDOW_SECONDS = 360 * DAY
PAIR_SPECS = (
    ("grid", "BTC-FDUSD", "BTCFDUSD", 200.0),
    ("grid", "ETH-FDUSD", "ETHFDUSD", 200.0),
    ("dca", "BTC-USDT", "BTCUSDT", 190.0),
    ("dca", "ETH-USDT", "ETHUSDT", 190.0),
)


def _resolve_release(root: Path, event: Mapping[str, Any]) -> Path:
    candidates = [
        root / "releases" / str(event.get("release_sha256", "")),
        root / "current",
        root,
    ]
    for candidate in candidates:
        if (candidate / "evidence/summary.json").is_file():
            return candidate
    raise FileNotFoundError("no release evidence/summary.json matches the notification")


def _read_summary(root: Path) -> dict[str, Any]:
    return json.loads((root / "evidence/summary.json").read_text(encoding="utf-8"))


def _series(root: Path, strategy: str, pair: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    last_kept = 0
    with gzip.open(root / "evidence/audit_series.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["strategy"] != strategy or row["pair"] != pair:
                continue
            timestamp = int(float(row["timestamp"]))
            if timestamp - last_kept < 3600:
                continue
            last_kept = timestamp
            rows.append({key: float(row[key]) for key in (
                "timestamp", "equity", "peak_equity", "drawdown_pct", "price",
                "probability", "entry_threshold",
            )})
    if not rows:
        raise ValueError(f"parameter evidence has no {strategy}/{pair} series")
    return rows


def _intervals(root: Path, strategy: str, pair: str) -> list[tuple[int, int]]:
    output = []
    with (root / "evidence/risk_intervals.csv").open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if (row.get("strategy") == strategy and row.get("pair") == pair
                    and row.get("mechanism") == "v22_weekly_buy_gate"
                    and row.get("phase") in {"RISK_OFF", "EXITING", "COOLDOWN", "REENTRY"}):
                output.append((int(float(row["start_ts"])), int(float(row["end_ts"]))))
    return output


def _binance_hourly(symbol: str, start: int, end: int) -> list[tuple[int, float]]:
    output: list[tuple[int, float]] = []
    cursor = start
    while cursor < end:
        response = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": "1h", "startTime": cursor * 1000,
                    "endTime": end * 1000, "limit": 1000}, timeout=25,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        output.extend((int(item[0] / 1000), float(item[4])) for item in batch)
        next_cursor = int(batch[-1][0] / 1000) + 3600
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return output


def _xy(values: list[tuple[float, float]], box: tuple[int, int, int, int],
        start: int, end: int) -> list[tuple[int, int]]:
    left, top, right, bottom = box
    finite = [value for _, value in values if math.isfinite(value)]
    if not finite:
        return []
    low, high = min(finite), max(finite)
    if high == low:
        low -= 1
        high += 1
    return [
        (left + int((timestamp - start) / max(1, end - start) * (right - left)),
         top + int((high - value) / (high - low) * (bottom - top)))
        for timestamp, value in values if start <= timestamp <= end and math.isfinite(value)
    ]


def _panel(draw: ImageDraw.ImageDraw, title: str, box: tuple[int, int, int, int],
           values: list[tuple[float, float]], start: int, end: int, color: str,
           intervals: list[tuple[int, int]], invalid_end: int) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=16, fill="#ffffff", outline="#dce3ea", width=2)
    draw.text((left + 24, top + 18), title, fill="#172033", font=report_font(29, bold=True))
    chart = (left + 26, top + 70, right - 26, bottom - 30)
    invalid_right = chart[0]
    if invalid_end > start:
        invalid_right = chart[0] + int((min(invalid_end, end) - start) / (end - start) * (chart[2] - chart[0]))
        draw.rectangle((chart[0], chart[1], invalid_right, chart[3]), fill="#fee2e2")
    for begin, finish in intervals:
        begin, finish = max(begin, start), min(finish, end)
        if finish <= begin:
            continue
        x0 = chart[0] + int((begin - start) / (end - start) * (chart[2] - chart[0]))
        x1 = chart[0] + int((finish - start) / (end - start) * (chart[2] - chart[0]))
        draw.rectangle((x0, chart[1], x1, chart[3]), fill="#fef3c7")
        draw.line((x0, chart[1], x0, chart[3]), fill="#b45309", width=2)
        draw.polygon(((x0, chart[1] + 4), (x0 - 8, chart[1] + 19),
                      (x0 + 8, chart[1] + 19)), fill="#b45309")
        draw.line((x1, chart[1], x1, chart[3]), fill="#15803d", width=2)
        draw.polygon(((x1, chart[1] + 19), (x1 - 8, chart[1] + 4),
                      (x1 + 8, chart[1] + 4)), fill="#15803d")
    if invalid_end > start:
        draw.rectangle((chart[0], chart[1], invalid_right, chart[3]),
                       outline="#dc2626", width=4)
    points = _xy(values, chart, start, end)
    if len(points) > 1:
        draw.line(points, fill=color, width=4, joint="curve")
    else:
        draw.text((chart[0] + 20, chart[1] + 45), "无可信数据", fill="#b91c1c", font=report_font(28))


def render_360_card(root: Path, strategy: str, pair: str, symbol: str,
                    initial_equity: float, target: Path) -> dict[str, Any]:
    rows = _series(root, strategy, pair)
    evidence_start = int(rows[0]["timestamp"])
    # Only complete UTC natural days are admissible in the 360-day evidence
    # window; an intraday model/evidence cutoff never becomes a partial day.
    end = int(rows[-1]["timestamp"]) // DAY * DAY
    start = end - WINDOW_SECONDS
    early = _binance_hourly(symbol, start, evidence_start)
    if not early or early[0][0] > start + 7200:
        raise ValueError(f"{pair} cannot backfill the complete 360-day price window")
    prices = [(ts, price) for ts, price in early] + [
        (row["timestamp"], row["price"]) for row in rows
    ]
    equity = [(start, initial_equity), (evidence_start - 1, initial_equity)] + [
        (row["timestamp"], row["equity"]) for row in rows
    ]
    drawdown = [(start, 0.0), (evidence_start - 1, 0.0)] + [
        (row["timestamp"], row["drawdown_pct"]) for row in rows
    ]
    probability = [(row["timestamp"], row["probability"]) for row in rows]
    threshold = [(row["timestamp"], row["entry_threshold"]) for row in rows]
    intervals = _intervals(root, strategy, pair)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f5f7fa")
    draw = ImageDraw.Draw(image)
    draw.text((65, 45), f"{strategy.upper()} {pair}｜360天 v22 审计",
              fill="#172033", font=report_font(46, bold=True))
    draw.text((65, 112), "黄色=v22 Risk-Off；红色=无签名模型，Fail-Closed纯现金",
              fill="#64748b", font=report_font(27))
    boxes = ((65, 180, 1375, 680), (65, 720, 1375, 1220),
             (65, 1260, 1375, 1740), (65, 1780, 1375, 2260))
    _panel(draw, "价格", boxes[0], prices, start, end, "#2563eb", intervals, evidence_start)
    _panel(draw, "单机器人连续权益", boxes[1], equity, start, end, "#16a34a", intervals, evidence_start)
    _panel(draw, "回撤（%）", boxes[2], drawdown, start, end, "#dc2626", intervals, evidence_start)
    _panel(draw, "v22概率（蓝）/逐周阈值（红）", boxes[3], probability, start, end,
           "#2563eb", intervals, evidence_start)
    threshold_points = _xy(threshold, (91, 1850, 1349, 2230), start, end)
    if len(threshold_points) > 1:
        draw.line(threshold_points, fill="#dc2626", width=3)
    draw.text((65, 2310), f"UTC {datetime.fromtimestamp(start, timezone.utc).date()} → "
              f"{datetime.fromtimestamp(end, timezone.utc).date()}｜不使用组合权益",
              fill="#64748b", font=report_font(27))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, "PNG", optimize=True)
    return {"start": start, "end": end, "evidence_start": evidence_start,
            "rows": len(rows), "png_sha256": sha256_file(target)}


def build_parameter_attachments(event: Mapping[str, Any], *, release_root: Path,
                                output_root: Path) -> list[dict[str, Any]]:
    request = str(event.get("details", {}).get("report_request", ""))
    if request not in {"v22_360d", "grid_360d"}:
        return []
    directory = output_root / str(event["event_id"])
    directory.mkdir(parents=True, exist_ok=True)
    pdf = directory / "parameter_analysis.pdf"
    report_id = str(event["event_id"])[:24]
    if request == "grid_360d":
        render_analysis_pdf({
            "title": "Grid 参数分析报告", "report_id": report_id,
            "status": event.get("transition"), "generated_at": event.get("occurred_at"),
            "parameter_version": event.get("details", {}).get("parameter_version"),
            "parameter_sha256": event.get("parameter_sha256"),
            "evidence_complete": False, "conclusion": event.get("reason"),
            "sections": [
                {"title": "旧参数", "text": json.dumps(
                    event.get("details", {}).get("previous_parameters", {}), ensure_ascii=False,
                )},
                {"title": "候选参数", "text": json.dumps(
                    event.get("details", {}).get("candidate", {}), ensure_ascii=False,
                )},
                {"title": "最佳候选与拒绝原因", "text": json.dumps({
                    "evaluation": event.get("details", {}).get(
                        "best_rejected_candidate",
                        event.get("details", {}).get("best_candidate_evaluation", {}),
                    ),
                    "reason": event.get("details", {}).get("rejection_reason", "已满足既有门槛"),
                }, ensure_ascii=False)},
                {"title": "证据状态", "text": (
                    "当前候选没有哈希匹配的360天连续回放，因此不生成PNG；"
                    "本缺失不改变原有参数部署门槛。"
                )},
            ],
        }, pdf)
        return [{"path": str(pdf), "kind": "document",
                 "sha256": sha256_file(pdf), "caption": "Grid参数分析PDF（360天证据缺失）",
                 "evidence_complete": False}]
    release_root = _resolve_release(release_root, event)
    summary = _read_summary(release_root)
    evidence_model = str(summary.get("frozen_inputs", {}).get("model_sha256", ""))
    requested_model = str(event.get("model_sha256", ""))
    if requested_model and requested_model != evidence_model:
        render_analysis_pdf({
            "title": "v22 参数分析报告", "report_id": report_id,
            "status": event.get("transition"), "generated_at": event.get("occurred_at"),
            "release_sha256": event.get("release_sha256"),
            "model_sha256": requested_model, "evidence_complete": False,
            "conclusion": "候选模型哈希没有匹配的360天回放证据",
            "sections": [{"title": "哈希核验", "text": (
                f"requested={requested_model}; evidence={evidence_model}。不生成PNG。"
            )}, {"title": "训练/校准/样本外窗口", "text": json.dumps(
                event.get("details", {}).get("weekly_report", {}), ensure_ascii=False,
            )}],
        }, pdf)
        return [{"path": str(pdf), "kind": "document", "sha256": sha256_file(pdf),
                 "caption": "v22参数分析PDF（模型证据不匹配）",
                 "evidence_complete": False}]
    attachments: list[dict[str, Any]] = []
    audits = []
    for strategy, pair, symbol, initial in PAIR_SPECS:
        target = directory / f"{strategy}_{pair.replace('-', '').lower()}_360d.png"
        audit = render_360_card(release_root, strategy, pair, symbol, initial, target)
        audits.append({"strategy": strategy, "pair": pair, **audit})
        attachments.append({"path": str(target), "kind": "photo",
                            "sha256": audit["png_sha256"],
                            "caption": f"{strategy.upper()} {pair}｜360天v22阴影审计"})
    render_analysis_pdf({
        "title": "ethbtc-forced-exit 参数分析报告", "report_id": report_id,
        "status": event.get("transition"), "generated_at": event.get("occurred_at"),
        "parameter_version": event.get("details", {}).get("parameter_version"),
        "release_sha256": event.get("release_sha256"),
        "model_sha256": requested_model or evidence_model, "evidence_complete": True,
        "conclusion": event.get("reason"),
        "sections": [
            {"title": "完整性", "text": "四个机器人分别计算，不合并权益。无签名模型区间按Fail-Closed纯现金展示。"},
            {"title": "训练/校准/样本外窗口", "text": json.dumps(
                event.get("details", {}).get("weekly_report", {}), ensure_ascii=False,
            )},
            {"title": "模型与数据哈希", "text": json.dumps({
                key: event.get("details", {}).get(key) for key in (
                    "feature_schema_sha256", "strategy_schema_sha256", "training_data_sha256",
                )
            }, ensure_ascii=False)},
            {"title": "360天回放", "text": json.dumps(audits, ensure_ascii=False)},
            {"title": "逐机器人收益/回撤/成交", "text": json.dumps(
                summary.get("metrics", []), ensure_ascii=False,
            )},
        ],
    }, pdf)
    attachments.insert(0, {"path": str(pdf), "kind": "document",
                           "sha256": sha256_file(pdf), "caption": "参数分析PDF"})
    return attachments
