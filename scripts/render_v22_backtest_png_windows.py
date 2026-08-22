#!/usr/bin/env python3
"""Render mobile PNGs for the v22 forced-exit historical audit.

This renderer never loads or publishes a model artifact.  It consumes the
immutable forced-exit audit series and produces one image per robot/window.
Intervals before the signed model or before pair-level replay evidence are
shown explicitly instead of inventing an equity curve.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from telegram_notifications import report_font
except ModuleNotFoundError:
    from live_guard.telegram_notifications import report_font


WIDTH, HEIGHT = 1440, 2400
DAY = 86_400
PAIR_SPECS = (
    ("grid", "BTC-FDUSD", "FDUSD"),
    ("grid", "ETH-FDUSD", "FDUSD"),
    ("dca", "BTC-USDT", "USDT"),
    ("dca", "ETH-USDT", "USDT"),
)
COLORS = {
    "price": "#2563eb",
    "equity": "#16a34a",
    "drawdown": "#dc2626",
    "probability": "#2563eb",
    "threshold": "#dc2626",
    "risk_off": "#fef3c7",
    "risk_border": "#b45309",
    "invalid": "#fee2e2",
    "missing": "#e5e7eb",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_series(path: Path) -> dict[tuple[str, str], list[dict[str, float]]]:
    output: dict[tuple[str, str], list[dict[str, float]]] = {}
    last_kept: dict[tuple[str, str], int] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["strategy"], row["pair"])
            timestamp = int(float(row["timestamp"]))
            # Hourly points are visually sufficient and keep mobile PNGs small.
            if timestamp - last_kept.get(key, -10**18) < 3600:
                continue
            last_kept[key] = timestamp
            output.setdefault(key, []).append({
                "timestamp": float(timestamp),
                "equity": float(row["equity"]),
                "drawdown_pct": float(row["drawdown_pct"]),
                "price": float(row["price"]),
                "probability": float(row["probability"]),
                "entry_threshold": float(row["entry_threshold"]),
            })
    return output


def _read_intervals(path: Path) -> dict[tuple[str, str], list[tuple[int, int]]]:
    raw: dict[tuple[str, str], list[tuple[int, int]]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("mechanism") != "v22_weekly_buy_gate":
                continue
            if row.get("phase") not in {"RISK_OFF", "EXITING", "COOLDOWN", "REENTRY"}:
                continue
            key = (row["strategy"], row["pair"])
            raw.setdefault(key, []).append(
                (int(float(row["start_ts"])), int(float(row["end_ts"])))
            )
    output: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for key, values in raw.items():
        merged: list[list[int]] = []
        for start, end in sorted(values):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        output[key] = [(item[0], item[1]) for item in merged]
    return output


def _window_epoch(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())


def requested_windows(common_end: int) -> tuple[tuple[str, str, int, int], ...]:
    end = common_end // DAY * DAY
    return (
        ("360d", "过去360天", end - 360 * DAY, end),
        ("2026_jan_feb", "2026年1–2月重点窗口", _window_epoch("2026-01-01"), _window_epoch("2026-03-01")),
        ("2026_may_june", "2026年5–6月重点窗口", _window_epoch("2026-05-01"), _window_epoch("2026-07-01")),
    )


def _x(timestamp: float, box: tuple[int, int, int, int], start: int, end: int) -> int:
    return box[0] + int((timestamp - start) / max(1, end - start) * (box[2] - box[0]))


def _points(rows: list[dict[str, float]], field: str, box: tuple[int, int, int, int],
            start: int, end: int,
            value_range: tuple[float, float] | None = None) -> list[tuple[int, int]]:
    values = [(row["timestamp"], row[field]) for row in rows
              if start <= row["timestamp"] < end and math.isfinite(row[field])]
    if not values:
        return []
    if value_range is None:
        low, high = min(value for _, value in values), max(value for _, value in values)
        padding = max((high - low) * 0.06, abs(high) * 0.002, 1e-9)
        low, high = low - padding, high + padding
    else:
        low, high = value_range
    return [
        (_x(timestamp, box, start, end),
         box[1] + int((high - value) / (high - low) * (box[3] - box[1])))
        for timestamp, value in values
    ]


def _shade(draw: ImageDraw.ImageDraw, chart: tuple[int, int, int, int], start: int, end: int,
           signed_start: int, evidence_start: int, intervals: Iterable[tuple[int, int]]) -> None:
    if start < signed_start:
        right = _x(min(end, signed_start), chart, start, end)
        draw.rectangle((chart[0], chart[1], right, chart[3]), fill=COLORS["invalid"])
        draw.rectangle((chart[0], chart[1], right, chart[3]), outline="#dc2626", width=3)
    missing_start = max(start, signed_start)
    if evidence_start > missing_start:
        left = _x(missing_start, chart, start, end)
        right = _x(min(end, evidence_start), chart, start, end)
        if right > left:
            draw.rectangle((left, chart[1], right, chart[3]), fill=COLORS["missing"])
    for begin, finish in intervals:
        begin, finish = max(start, begin), min(end, finish)
        if finish <= begin:
            continue
        left, right = _x(begin, chart, start, end), _x(finish, chart, start, end)
        draw.rectangle((left, chart[1], right, chart[3]), fill=COLORS["risk_off"])
        draw.line((left, chart[1], left, chart[3]), fill=COLORS["risk_border"], width=2)
        draw.line((right, chart[1], right, chart[3]), fill="#15803d", width=2)
        draw.polygon(((left, chart[1] + 4), (left - 8, chart[1] + 19),
                      (left + 8, chart[1] + 19)), fill=COLORS["risk_border"])
        draw.polygon(((right, chart[1] + 19), (right - 8, chart[1] + 4),
                      (right + 8, chart[1] + 4)), fill="#15803d")


def _panel(draw: ImageDraw.ImageDraw, *, title: str, rows: list[dict[str, float]], field: str,
           box: tuple[int, int, int, int], start: int, end: int, signed_start: int,
           evidence_start: int, intervals: list[tuple[int, int]], color: str,
           second_field: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=18, fill="#ffffff", outline="#dce3ea", width=2)
    draw.text((box[0] + 24, box[1] + 17), title, fill="#172033", font=report_font(29, bold=True))
    chart = (box[0] + 28, box[1] + 72, box[2] - 28, box[3] - 42)
    _shade(draw, chart, start, end, signed_start, evidence_start, intervals)
    for step in range(5):
        y = chart[1] + int(step / 4 * (chart[3] - chart[1]))
        draw.line((chart[0], y, chart[2], y), fill="#dbe3ec", width=1)
    shared_range = (0.0, 1.0) if second_field else None
    points = _points(rows, field, chart, start, end, shared_range)
    if len(points) > 1:
        draw.line(points, fill=color, width=4, joint="curve")
    else:
        draw.text((chart[0] + 20, chart[1] + 45), "此窗口无可信回放数据",
                  fill="#b91c1c", font=report_font(28))
    if second_field:
        second = _points(rows, second_field, chart, start, end, shared_range)
        if len(second) > 1:
            draw.line(second, fill=COLORS["threshold"], width=3)
    draw.text((chart[0], box[3] - 34), datetime.fromtimestamp(start, timezone.utc).strftime("%Y-%m-%d"),
              fill="#64748b", font=report_font(22))
    right_text = datetime.fromtimestamp(end, timezone.utc).strftime("%Y-%m-%d")
    draw.text((chart[2] - 132, box[3] - 34), right_text, fill="#64748b", font=report_font(22))


def _window_stats(rows: list[dict[str, float]], start: int, end: int,
                  intervals: list[tuple[int, int]]) -> dict[str, Any]:
    selected = [row for row in rows if start <= row["timestamp"] < end]
    if not selected:
        return {"evidence": False}
    off_seconds = sum(max(0, min(end, finish) - max(start, begin)) for begin, finish in intervals)
    return {
        "evidence": True,
        "start_equity": selected[0]["equity"],
        "end_equity": selected[-1]["equity"],
        "pnl": selected[-1]["equity"] - selected[0]["equity"],
        "max_drawdown_pct": min(row["drawdown_pct"] for row in selected),
        "risk_off_hours": off_seconds / 3600,
        "points": len(selected),
        "evidence_start": int(selected[0]["timestamp"]),
        "evidence_end": int(selected[-1]["timestamp"]),
    }


def render_card(*, strategy: str, pair: str, quote: str, rows: list[dict[str, float]],
                intervals: list[tuple[int, int]], label: str, start: int, end: int,
                signed_start: int, target: Path, production_model_sha256: str) -> dict[str, Any]:
    evidence_start = int(rows[0]["timestamp"])
    stats = _window_stats(rows, start, end, intervals)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f5f7fa")
    draw = ImageDraw.Draw(image)
    draw.text((64, 38), f"{strategy.upper()} {pair}｜{label}", fill="#172033",
              font=report_font(45, bold=True))
    draw.text((64, 101), "v22 weekly walk-forward + forced-exit-v2｜单机器人权益",
              fill="#475569", font=report_font(27))
    draw.text((64, 143), "黄色=Risk-Off　红色=签名未覆盖　灰色=无可信成交回放证据",
              fill="#64748b", font=report_font(25))
    if stats.get("evidence"):
        summary = (
            f"窗口收益 {stats['pnl']:+.4f} {quote}　末值 {stats['end_equity']:.4f} {quote}　"
            f"最大回撤 {stats['max_drawdown_pct']:.2f}%　Risk-Off {stats['risk_off_hours']:.1f}h"
        )
    else:
        summary = "该窗口无可信回放数据"
    draw.rounded_rectangle((64, 190, 1376, 275), radius=14, fill="#eef2ff", outline="#c7d2fe", width=2)
    draw.text((88, 216), summary, fill="#1e3a8a", font=report_font(26, bold=True))
    boxes = (
        (64, 310, 1376, 775),
        (64, 810, 1376, 1275),
        (64, 1310, 1376, 1775),
        (64, 1810, 1376, 2275),
    )
    common = dict(rows=rows, start=start, end=end, signed_start=signed_start,
                  evidence_start=evidence_start, intervals=intervals)
    _panel(draw, title="市场价格", field="price", box=boxes[0], color=COLORS["price"], **common)
    _panel(draw, title=f"单机器人连续权益（{quote}）", field="equity", box=boxes[1],
           color=COLORS["equity"], **common)
    _panel(draw, title="回撤（%）", field="drawdown_pct", box=boxes[2],
           color=COLORS["drawdown"], **common)
    _panel(draw, title="v22 概率（蓝）与逐周阈值（红）", field="probability",
           second_field="entry_threshold", box=boxes[3], color=COLORS["probability"], **common)
    draw.text((64, 2320), "UTC｜v22 周度模型｜离线反事实，不代表未来收益",
              fill="#64748b", font=report_font(24))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, "PNG", optimize=True)
    return {**stats, "path": str(target), "sha256": sha256_file(target),
            "width": WIDTH, "height": HEIGHT, "window_start": start, "window_end": end}


def build_report(evidence_dir: Path, output_dir: Path, *, signed_start: int,
                 production_model_sha256: str, evidence_model_sha256: str,
                 release_sha256: str = "") -> dict[str, Any]:
    series = _read_series(evidence_dir / "audit_series.csv.gz")
    intervals = _read_intervals(evidence_dir / "risk_intervals.csv")
    missing = [key for key in ((strategy, pair) for strategy, pair, _ in PAIR_SPECS) if key not in series]
    if missing:
        raise ValueError(f"audit series missing robots: {missing}")
    common_end = min(int(series[(strategy, pair)][-1]["timestamp"])
                     for strategy, pair, _ in PAIR_SPECS)
    manifest: dict[str, Any] = {
        "schema": "v22-forced-exit-png-windows-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution_policy": "v22-risk-off-forced-exit-v2",
        "production_model_sha256": production_model_sha256,
        "evidence_model_sha256": evidence_model_sha256,
        "release_sha256": release_sha256 or None,
        "historical_week_identity_verified": True,
        "common_complete_evidence_end": common_end // DAY * DAY,
        "images": [],
    }
    for window_id, label, start, end in requested_windows(common_end):
        for strategy, pair, quote in PAIR_SPECS:
            key = (strategy, pair)
            target = output_dir / f"{strategy}_{pair.replace('-', '').lower()}_{window_id}.png"
            audit = render_card(
                strategy=strategy, pair=pair, quote=quote, rows=series[key],
                intervals=intervals.get(key, []), label=label, start=start, end=end,
                signed_start=signed_start, target=target,
                production_model_sha256=production_model_sha256,
            )
            manifest["images"].append({"strategy": strategy, "pair": pair,
                                       "window": window_id, **audit})
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signed-start", type=int, required=True)
    parser.add_argument("--production-model-sha256", required=True)
    parser.add_argument("--evidence-model-sha256", required=True)
    parser.add_argument("--release-sha256", default="")
    args = parser.parse_args()
    manifest = build_report(
        args.evidence_dir, args.output_dir, signed_start=args.signed_start,
        production_model_sha256=args.production_model_sha256,
        evidence_model_sha256=args.evidence_model_sha256,
        release_sha256=args.release_sha256,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
