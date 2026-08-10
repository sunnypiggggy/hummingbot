from pathlib import Path

from PIL import Image

from scripts.render_v22_backtest_png_windows import requested_windows, render_card


def test_requested_windows_include_exact_focus_periods() -> None:
    windows = {item[0]: item for item in requested_windows(1785117300)}
    assert windows["2026_jan_feb"][2:] == (1767225600, 1772323200)
    assert windows["2026_may_june"][2:] == (1777593600, 1782864000)
    assert windows["360d"][3] - windows["360d"][2] == 360 * 86400


def test_render_card_is_mobile_and_pair_specific(tmp_path: Path) -> None:
    start = 1767225600
    rows = [
        {"timestamp": float(start + hour * 3600), "equity": 190 + hour / 100,
         "drawdown_pct": -hour / 1000, "price": 100 + hour,
         "probability": 0.1 + hour / 1000, "entry_threshold": 0.2}
        for hour in range(72)
    ]
    target = tmp_path / "dca_btc.png"
    result = render_card(
        strategy="dca", pair="BTC-USDT", quote="USDT", rows=rows,
        intervals=[(start + 10 * 3600, start + 20 * 3600)],
        label="测试窗口", start=start, end=start + 72 * 3600,
        signed_start=start, target=target, production_model_sha256="a" * 64,
    )
    with Image.open(target) as image:
        assert image.size == (1440, 2400)
    assert result["pnl"] > 0
    assert result["risk_off_hours"] == 10
