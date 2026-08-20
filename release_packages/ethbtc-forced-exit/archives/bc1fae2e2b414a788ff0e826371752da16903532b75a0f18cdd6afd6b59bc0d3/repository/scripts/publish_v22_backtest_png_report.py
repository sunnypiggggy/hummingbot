#!/usr/bin/env python3
"""Build a Telegram event containing only v22 backtest PNG attachments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_guard.telegram_notifications import build_event, sha256_file


WINDOW_LABELS = {
    "360d": "过去360天",
    "2026_jan_feb": "2026年1–2月重点窗口",
    "2026_may_june": "2026年5–6月重点窗口",
}


def build_png_event(manifest_path: Path, *, attachment_root: str,
                    release_sha256: str) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(manifest_path)
    images = list(manifest.get("images", []))
    identities = {(item.get("strategy"), item.get("pair"), item.get("window")) for item in images}
    expected = {
        (strategy, pair, window)
        for strategy, pair in (("grid", "BTC-FDUSD"), ("grid", "ETH-FDUSD"),
                               ("dca", "BTC-USDT"), ("dca", "ETH-USDT"))
        for window in WINDOW_LABELS
    }
    if identities != expected or len(images) != 12:
        raise ValueError("PNG manifest must contain exactly four robots x three windows")
    attachments = []
    for item in images:
        local = Path(item["path"])
        if not local.is_file() or sha256_file(local) != item["sha256"]:
            raise ValueError(f"PNG missing or hash mismatch: {local}")
        # /workspace/releases is the ethbtc-forced-exit package root in the
        # report container, not the parent release_packages directory.
        relative = local.relative_to(manifest_path.parents[2]).as_posix()
        attachments.append({
            "path": f"{attachment_root.rstrip('/')}/{relative}",
            "kind": "photo",
            "sha256": item["sha256"],
            "caption": f"{item['strategy'].upper()} {item['pair']}｜{WINDOW_LABELS[item['window']]}｜单机器人权益",
        })
    model_sha = str(manifest["production_model_sha256"])
    start = next(item["window_start"] for item in images if item["window"] == "360d")
    end = next(item["window_end"] for item in images if item["window"] == "360d")
    reason = (
        "已按当前生产 v22 历史周信号与 forced-exit-v2 执行口径生成回测图片："
        "Grid BTC/ETH、DCA BTC/ETH 各自独立权益，包含过去360天、2026年1–2月和2026年5–6月。"
        "当前生产包与回放证据覆盖的历史周模型、阈值和边界已逐项一致性核验；"
        "签名未覆盖或无可信成交证据的区间在图中单独标示。此消息只发送PNG，不附模型文件。"
    )
    return build_event(
        source="v22-backtest-png-report",
        strategy="grid+dca",
        bot="grid-live-fdusd-400,dca-live-btcusdt-200,dca-live-ethusdt-200",
        pair="BTC-FDUSD,ETH-FDUSD,BTC-USDT,ETH-USDT",
        mechanism="parameter_update",
        transition="PARAMETER_RETAINED",
        reason=reason,
        severity="info",
        action="publish_backtest_png_only",
        release_sha256=release_sha256,
        model_sha256=model_sha,
        correlation_id=f"v22-backtest-png:{release_sha256}:{manifest_hash}",
        attachments=attachments,
        details={
            "report_schema": manifest["schema"],
            "execution_policy": manifest["execution_policy"],
            "historical_week_identity_verified": manifest["historical_week_identity_verified"],
            "evidence_model_sha256": manifest["evidence_model_sha256"],
            "production_model_sha256": model_sha,
            "window_360d_start": start,
            "window_360d_end": end,
            "image_count": len(attachments),
            "model_attachment_included": False,
            "images": [{key: item.get(key) for key in (
                "strategy", "pair", "window", "pnl", "end_equity",
                "max_drawdown_pct", "risk_off_hours", "sha256",
            )} for item in images],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attachment-root", default="/workspace/releases")
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    event = build_png_event(args.manifest, attachment_root=args.attachment_root,
                            release_sha256=args.release_sha256)
    args.output.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event_id": event["event_id"], "photos": len(event["attachments"]),
                      "documents": sum(item["kind"] == "document" for item in event["attachments"])},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
