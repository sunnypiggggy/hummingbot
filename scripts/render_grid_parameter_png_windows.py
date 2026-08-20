#!/usr/bin/env python3
"""Render hash-bound mobile evidence for a Grid parameter candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1440, 2400
DAY = 86_400
WINDOWS = (
    ("360d", "过去360天", 1_755_993_600, 1_787_097_600),
    ("2026_jan_feb", "2026年1–2月重点窗口", 1_767_225_600, 1_772_323_200),
    ("2026_may_june", "2026年5–6月重点窗口", 1_777_593_600, 1_782_864_000),
)
PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
BASELINE_ARM = "live_fixed_5_25"
CANDIDATE_ARM = "btc_medium_eth_long"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold
        else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def points(values: Iterable[tuple[float, float]], box: tuple[int, int, int, int],
           start: int, end: int, *, y_range: tuple[float, float] | None = None) -> list[tuple[int, int]]:
    source = [(float(ts), float(value)) for ts, value in values
              if start <= float(ts) < end and math.isfinite(float(value))]
    if not source:
        return []
    low, high = y_range or (min(value for _, value in source), max(value for _, value in source))
    if high <= low:
        low, high = low - 1.0, high + 1.0
    left, top, right, bottom = box
    return [
        (left + int((ts - start) / max(1, end - start) * (right - left)),
         top + int((high - value) / (high - low) * (bottom - top)))
        for ts, value in source
    ]


def intervals(frame: pd.DataFrame, start: int, end: int) -> list[tuple[int, int, str]]:
    rows = frame[(frame.timestamp >= start) & (frame.timestamp < end)][["timestamp", "v22_state"]]
    output: list[tuple[int, int, str]] = []
    active: tuple[int, str] | None = None
    for row in rows.itertuples(index=False):
        state = str(row.v22_state)
        shaded = state if state != "RISK_ON" else ""
        if active and active[1] != shaded:
            output.append((active[0], int(row.timestamp), active[1]))
            active = None
        if shaded and active is None:
            active = (int(row.timestamp), shaded)
    if active:
        output.append((active[0], end, active[1]))
    return output


def draw_panel(draw: ImageDraw.ImageDraw, title: str, box: tuple[int, int, int, int],
               series: list[tuple[str, str, list[tuple[float, float]]]],
               risk_regions: list[tuple[int, int, str]], start: int, end: int) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, fill="#ffffff", outline="#d7dee8", width=2)
    draw.text((left + 24, top + 16), title, fill="#172033", font=font(28, bold=True))
    chart = (left + 30, top + 74, right - 30, bottom - 34)
    for begin, finish, state in risk_regions:
        x0 = chart[0] + int((max(begin, start) - start) / (end - start) * (chart[2] - chart[0]))
        x1 = chart[0] + int((min(finish, end) - start) / (end - start) * (chart[2] - chart[0]))
        if x1 <= x0:
            continue
        fill = "#fee2e2" if state == "UNAVAILABLE" else "#fef3c7"
        outline = "#dc2626" if state == "UNAVAILABLE" else "#b45309"
        draw.rectangle((x0, chart[1], x1, chart[3]), fill=fill, outline=outline, width=2)
    all_values = [value for _, _, values in series for ts, value in values
                  if start <= ts < end and math.isfinite(value)]
    y_range = (min(all_values), max(all_values)) if all_values else None
    legend_x = max(left + 620, right - len(series) * 270)
    for label, color, values in series:
        line = points(values, chart, start, end, y_range=y_range)
        if len(line) > 1:
            draw.line(line, fill=color, width=4, joint="curve")
        draw.line((legend_x, top + 34, legend_x + 42, top + 34), fill=color, width=5)
        draw.text((legend_x + 52, top + 17), label, fill="#475569", font=font(22))
        legend_x += 270
    if not all_values:
        draw.text((chart[0] + 20, chart[1] + 50), "无可信数据", fill="#b91c1c", font=font(28))


def render_pair_window(frame: pd.DataFrame, pair: str, window_id: str, label: str,
                       start: int, end: int, target: Path, parameter_sha256: str) -> dict[str, Any]:
    pair_frame = frame[frame.pair == pair].copy()
    baseline = pair_frame[pair_frame.arm == BASELINE_ARM].sort_values("timestamp")
    candidate = pair_frame[pair_frame.arm == CANDIDATE_ARM].sort_values("timestamp")
    if baseline.empty or candidate.empty:
        raise ValueError(f"missing baseline/candidate evidence for {pair}")
    risk_regions = intervals(candidate, start, end)
    base_peak = baseline.equity.cummax()
    candidate_peak = candidate.equity.cummax()
    base_dd = (baseline.equity / base_peak - 1.0) * 100.0
    candidate_dd = (candidate.equity / candidate_peak - 1.0) * 100.0
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f3f6fa")
    draw = ImageDraw.Draw(image)
    profile = "中短期横盘" if pair.startswith("BTC") else "长期波动"
    draw.text((64, 42), f"GRID {pair}｜{label}", fill="#172033", font=font(43, bold=True))
    draw.text((64, 104), f"候选={profile}｜对照=现网固定6%｜参数哈希 {parameter_sha256[:16]}…",
              fill="#475569", font=font(25))
    draw.text((64, 148), "黄色=v22 Risk-Off；红色=模型不可用（Fail-Closed）；FOMC不参与历史回放",
              fill="#64748b", font=font(24))
    boxes = ((64, 215, 1376, 680), (64, 720, 1376, 1245),
             (64, 1285, 1376, 1780), (64, 1820, 1376, 2280))
    draw_panel(draw, "标的价格", boxes[0], [
        ("价格", "#2563eb", list(zip(candidate.timestamp, candidate.price))),
    ], risk_regions, start, end)
    draw_panel(draw, "单交易对连续权益（FDUSD）", boxes[1], [
        ("固定参数", "#64748b", list(zip(baseline.timestamp, baseline.equity))),
        ("候选参数", "#d97706", list(zip(candidate.timestamp, candidate.equity))),
    ], risk_regions, start, end)
    draw_panel(draw, "峰值回撤（%）", boxes[2], [
        ("固定参数", "#64748b", list(zip(baseline.timestamp, base_dd))),
        ("候选参数", "#dc2626", list(zip(candidate.timestamp, candidate_dd))),
    ], risk_regions, start, end)
    base_lookup = baseline.set_index("timestamp").equity
    candidate_lookup = candidate.set_index("timestamp").equity
    common = base_lookup.index.intersection(candidate_lookup.index)
    delta = candidate_lookup.loc[common] - base_lookup.loc[common]
    draw_panel(draw, "候选相对固定参数权益差（FDUSD）", boxes[3], [
        ("权益差", "#16a34a", list(zip(common, delta))),
    ], risk_regions, start, end)
    draw.text((64, 2330),
              f"UTC {datetime.fromtimestamp(start, timezone.utc).date()} → "
              f"{datetime.fromtimestamp(end, timezone.utc).date()}｜离线证据，不代表自动授权",
              fill="#64748b", font=font(24))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, "PNG", optimize=True)
    return {
        "strategy": "grid", "pair": pair, "window": window_id,
        "window_start": start, "window_end": end, "path": target.name,
        "width": WIDTH, "height": HEIGHT, "sha256": sha256_file(target),
    }


def build_report(source: Path, output: Path, parameter_sha256: str) -> dict[str, Any]:
    if len(parameter_sha256) != 64:
        raise ValueError("parameter_sha256 must be a full SHA-256")
    equity_path = source / "continuous_equity.csv.gz"
    frame = pd.read_csv(equity_path)
    output.mkdir(parents=True, exist_ok=True)
    images = []
    for pair in PAIRS:
        prefix = pair.split("-")[0].lower()
        for window_id, label, start, end in WINDOWS:
            images.append(render_pair_window(
                frame, pair, window_id, label, start, end,
                output / f"grid_{prefix}_{window_id}.png", parameter_sha256,
            ))
    manifest = {
        "schema": "grid-parameter-mobile-evidence-v1",
        "parameter_sha256": parameter_sha256,
        "source_equity_sha256": sha256_file(equity_path),
        "baseline_arm": BASELINE_ARM, "candidate_arm": CANDIDATE_ARM,
        "images": images, "evidence_complete": len(images) == 6,
    }
    path = output / "grid_parameter_evidence_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameter-sha256", required=True)
    args = parser.parse_args()
    result = build_report(args.source.resolve(), args.output.resolve(), args.parameter_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
