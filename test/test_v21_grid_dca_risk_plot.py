from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_v21_grid_dca_risk import MECHANISMS, build_figure, render_dashboard  # noqa: E402


def _series() -> pd.DataFrame:
    rows = []
    for strategy, pairs in {
        "grid": ("BTC-FDUSD", "ETH-FDUSD"),
        "dca": ("BTC-USDT", "ETH-USDT"),
    }.items():
        for pair in pairs:
            for ts, price in ((1000, 100.0), (2000, 99.0), (3000, 101.0)):
                rows.append({"strategy": strategy, "pair": pair, "timestamp": ts,
                             "price": price, "equity": 200 + price / 100})
    return pd.DataFrame(rows)


def _intervals() -> pd.DataFrame:
    return pd.DataFrame([
        {"strategy": strategy, "pair": pair, "mechanism": mechanism,
         "start_ts": 1200, "end_ts": 2200, "trigger_value": "1",
         "threshold": "1", "action": "pause", "source": "test", "enabled": True}
        for strategy, pair in (("grid", "BTC-FDUSD"), ("dca", "BTC-USDT"))
        for mechanism in MECHANISMS
    ])


def test_each_risk_mechanism_has_independent_trace_and_shape_groups(tmp_path):
    series, intervals = _series(), _intervals()
    for strategy in ("grid", "dca"):
        figure, groups = build_figure(strategy, series, intervals)
        assert set(groups) == set(MECHANISMS)
        assert all(groups[name]["traces"] for name in MECHANISMS)
        assert all(groups[name]["shapes"] for name in MECHANISMS)
        assert len(figure.layout.shapes) == len(MECHANISMS)
    output = tmp_path / "risk.html"
    render_dashboard(series, intervals, output)
    html = output.read_text(encoding="utf-8")
    assert "Grid 与 DCA 七层风控" in html
    assert all(f"data-mechanism='{name}'" in html for name in MECHANISMS)
