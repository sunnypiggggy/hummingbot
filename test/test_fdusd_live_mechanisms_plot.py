from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_fdusd_live_mechanisms_1_3 import DEFAULT_RESULTS, build_figure, read_fomc  # noqa: E402


def test_plot_has_three_independent_mechanism_groups() -> None:
    figure, groups, counts = build_figure(DEFAULT_RESULTS, read_fomc(None))
    assert set(groups) == {"v21", "fomc", "pair_breaker", "shapes"}
    assert set(groups["shapes"]) == {"v21", "fomc", "pair_breaker"}
    assert counts["v21_intervals"] > 0
    assert counts["pair_breaker_intervals"] == 4
    assert counts["fomc_intervals"] == 0
    assert len(figure.layout.shapes) == sum(len(v) for v in groups["shapes"].values())
