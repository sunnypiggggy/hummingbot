"""Mobile PNG renderer for hash-bound Grid and v22 update evidence.

Parameter and model update reports intentionally emit photos only.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests
from PIL import Image, ImageDraw

try:
    from runtime_endpoints import binance_api_base
    from telegram_notifications import (
        report_font, sha256_file,
    )
except ModuleNotFoundError:
    from live_guard.runtime_endpoints import binance_api_base
    from live_guard.telegram_notifications import (
        report_font, sha256_file,
    )

try:
    from render_v22_backtest_png_windows import build_report as build_v22_png_windows
except ModuleNotFoundError:
    from scripts.render_v22_backtest_png_windows import build_report as build_v22_png_windows


WIDTH, HEIGHT = 1440, 2400
DAY = 86400
WINDOW_SECONDS = 360 * DAY
PAIR_SPECS = (
    ("grid", "BTC-FDUSD", "BTCFDUSD", 200.0),
    ("grid", "ETH-FDUSD", "ETHFDUSD", 200.0),
    ("dca", "BTC-USDT", "BTCUSDT", 190.0),
    ("dca", "ETH-USDT", "ETHUSDT", 190.0),
)


def _resolve_report_inputs(root: Path, event: Mapping[str, Any]) -> tuple[Path, Path]:
    """Resolve candidate identity separately from lineage replay evidence.

    A weekly candidate intentionally contains the signed model, policy and
    documentation, while the immutable 360-day replay evidence is stored once
    at the release-family root.  Treating them as one directory prevented a
    not-yet-active candidate from producing its approval charts.
    """
    release_sha = str(event.get("release_sha256", ""))
    requested = root / "releases" / release_sha
    identity_candidates = (requested, root / "current", root)
    identity = next(
        (
            candidate
            for candidate in identity_candidates
            if (
                (candidate / "shadow_package/shadow_lock.json").is_file()
                or (
                    candidate
                    / "inputs/frozen_v22/shadow_package/shadow_lock.json"
                ).is_file()
            )
        ),
        None,
    )
    if identity is None:
        raise FileNotFoundError(
            f"signed model identity is missing for approval release {release_sha or '-'}"
        )

    evidence_candidates = (requested, root, root / "current")
    evidence = next(
        (
            candidate
            for candidate in evidence_candidates
            if (
                (candidate / "evidence/summary.json").is_file()
                and (candidate / "evidence/audit_series.csv.gz").is_file()
                and (candidate / "evidence/risk_intervals.csv").is_file()
            )
        ),
        None,
    )
    if evidence is None:
        raise FileNotFoundError(
            "release-family replay evidence is incomplete: summary, audit series, "
            "and risk intervals are required"
        )
    return identity, evidence


def _read_summary(root: Path) -> dict[str, Any]:
    return json.loads((root / "evidence/summary.json").read_text(encoding="utf-8"))


def _signed_start(root: Path) -> int:
    candidates = (
        root / "shadow_package/shadow_lock.json",
        root / "inputs/frozen_v22/shadow_package/shadow_lock.json",
    )
    for path in candidates:
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            return int(value["effective_start"])
    raise FileNotFoundError("v22 signed effective_start is missing")


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
            f"{binance_api_base()}/api/v3/klines",
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
    if request not in {"v22_png_windows", "v22_360d", "grid_360d", "sol_grid_360d"}:
        return []
    directory = output_root / str(event["event_id"])
    directory.mkdir(parents=True, exist_ok=True)
    if request == "sol_grid_360d":
        evidence_root = Path(str(
            event.get("details", {}).get("evidence_root")
            or os.getenv("SOL_GRID_EVIDENCE_ROOT", "/workspace/sol-grid-evidence")
        ))
        manifest_path = evidence_root / "sol_grid_evidence_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"SOL PNG evidence manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "sol-grid-mobile-evidence-v1":
            raise ValueError("invalid SOL Grid PNG evidence schema")
        expected_identity = str(event.get("release_sha256") or event.get("model_sha256") or "")
        if not expected_identity or expected_identity not in {
            str(manifest.get("identity_sha256", "")), str(manifest.get("model_sha256", "")),
        }:
            raise ValueError("SOL PNG evidence is not bound to the requested candidate")
        images = list(manifest.get("images", []))
        if manifest.get("evidence_complete") is not True or len(images) != 3:
            raise ValueError("SOL update requires three complete PNG windows")
        labels = {
            "360d": "过去360天", "2026_jan_feb": "2026年1–2月重点窗口",
            "2026_may_june": "2026年5–6月重点窗口",
        }
        photos = []
        for row in images:
            image_path = evidence_root / str(row["path"])
            if not image_path.is_file() or sha256_file(image_path) != str(row["sha256"]):
                raise ValueError(f"SOL PNG evidence hash mismatch: {image_path.name}")
            with Image.open(image_path) as image:
                if image.size != (WIDTH, HEIGHT):
                    raise ValueError(f"SOL PNG is not mobile 1440x2400: {image_path.name}")
            photos.append({
                "path": str(image_path), "kind": "photo", "sha256": str(row["sha256"]),
                "caption": f"GRID SOL-FDUSD｜{labels[str(row['window'])]}｜短期/中短期横盘对照",
                "evidence_complete": True,
            })
        return photos
    if request == "grid_360d":
        evidence_root = Path(str(
            event.get("details", {}).get("evidence_root")
            or os.getenv("GRID_PARAMETER_EVIDENCE_ROOT", "/workspace/grid-parameter-evidence")
        ))
        manifest_path = evidence_root / "grid_parameter_evidence_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Grid PNG evidence manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "grid-parameter-mobile-evidence-v1":
            raise ValueError("invalid Grid PNG evidence schema")
        expected_parameter = str(event.get("parameter_sha256", ""))
        if not expected_parameter or manifest.get("parameter_sha256") != expected_parameter:
            raise ValueError("Grid PNG evidence is not bound to the requested parameter hash")
        images = list(manifest.get("images", []))
        if manifest.get("evidence_complete") is not True or len(images) != 6:
            raise ValueError("Grid update requires BTC/ETH x three complete PNG windows")
        labels = {
            "360d": "过去360天",
            "2026_jan_feb": "2026年1–2月重点窗口",
            "2026_may_june": "2026年5–6月重点窗口",
        }
        photos: list[dict[str, Any]] = []
        for row in images:
            image_path = evidence_root / str(row["path"])
            if not image_path.is_file() or sha256_file(image_path) != str(row["sha256"]):
                raise ValueError(f"Grid PNG evidence hash mismatch: {image_path.name}")
            with Image.open(image_path) as image:
                if image.size != (WIDTH, HEIGHT):
                    raise ValueError(f"Grid PNG is not mobile 1440x2400: {image_path.name}")
            photos.append({
                "path": str(image_path), "kind": "photo", "sha256": str(row["sha256"]),
                "caption": f"GRID {row['pair']}｜{labels[str(row['window'])]}｜参数候选对照证据",
                "evidence_complete": True,
            })
        return photos
    identity_root, evidence_root = _resolve_report_inputs(release_root, event)
    summary = _read_summary(evidence_root)
    evidence_model = str(summary.get("frozen_inputs", {}).get("model_sha256", ""))
    requested_model = str(event.get("model_sha256", ""))
    manifest = build_v22_png_windows(
        evidence_root / "evidence", directory,
        signed_start=_signed_start(identity_root),
        production_model_sha256=requested_model or evidence_model,
        evidence_model_sha256=evidence_model,
        release_sha256=str(event.get("release_sha256") or ""),
    )
    labels = {
        "360d": "过去360天",
        "2026_jan_feb": "2026年1–2月重点窗口",
        "2026_may_june": "2026年5–6月重点窗口",
    }
    attachments: list[dict[str, Any]] = []
    for image in manifest["images"]:
        attachments.append({
            "path": image["path"],
            "kind": "photo",
            "sha256": image["sha256"],
            "caption": (
                f"{image['strategy'].upper()} {image['pair']}｜"
                f"{labels[image['window']]}｜单机器人权益"
            ),
            "evidence_complete": bool(image.get("evidence")),
        })
    if len(attachments) != 12 or any(item["kind"] != "photo" for item in attachments):
        raise RuntimeError("v22 update report must contain four robots x three PNG windows")
    return attachments
