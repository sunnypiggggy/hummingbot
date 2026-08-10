import json
from pathlib import Path

from scripts.publish_v22_backtest_png_report import build_png_event


def test_event_contains_only_twelve_photos(tmp_path: Path) -> None:
    audit = tmp_path / "release_packages" / "ethbtc-forced-exit" / "audits" / "report"
    audit.mkdir(parents=True)
    images = []
    for strategy, pair in (("grid", "BTC-FDUSD"), ("grid", "ETH-FDUSD"),
                           ("dca", "BTC-USDT"), ("dca", "ETH-USDT")):
        for window in ("360d", "2026_jan_feb", "2026_may_june"):
            path = audit / f"{strategy}_{pair}_{window}.png"
            path.write_bytes(f"{strategy}:{pair}:{window}".encode())
            from live_guard.telegram_notifications import sha256_file
            images.append({"strategy": strategy, "pair": pair, "window": window,
                           "path": str(path), "sha256": sha256_file(path),
                           "window_start": 1, "window_end": 2})
    manifest = {"schema": "v22-forced-exit-png-windows-v1", "images": images,
                "production_model_sha256": "a" * 64, "evidence_model_sha256": "b" * 64,
                "execution_policy": "v22-risk-off-forced-exit-v2",
                "historical_week_identity_verified": True, "manifest_sha256": "c" * 64}
    manifest_path = audit / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    event = build_png_event(manifest_path, attachment_root="/workspace/releases", release_sha256="d" * 64)
    assert len(event["attachments"]) == 12
    assert {item["kind"] for item in event["attachments"]} == {"photo"}
    assert event["details"]["model_attachment_included"] is False
