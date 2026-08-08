from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")
sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts").resolve()))

from deep_learning_long_risk_models_v12 import (  # noqa: E402
    DualBranchLongRiskModel, RobustSequenceScaler, deterministic_configurations,
    seed_everything,
)


OUT = Path("results/backtests/deep_learning_long_risk_gate_v12")


def test_24_deterministic_configurations_cover_all_architectures():
    first = deterministic_configurations()
    second = deterministic_configurations()
    assert first == second
    assert len(first) == 24
    assert len({item["config_id"] for item in first}) == 24
    assert {item["architecture"] for item in first} == {"tcn", "gru", "transformer"}
    assert all(sum(item["architecture"] == architecture for item in first) == 8
               for architecture in ("tcn", "gru", "transformer"))


@pytest.mark.parametrize("architecture", ["tcn", "gru", "transformer"])
def test_dual_branch_model_outputs_two_finite_logits(architecture: str):
    seed_everything(42, 1)
    config = next(item for item in deterministic_configurations() if item["architecture"] == architecture)
    model = DualBranchLongRiskModel(6, 4, config)
    output = model(torch.zeros((3, 168, 6)), torch.zeros((3, 288, 4)))
    assert output.shape == (3, 2)
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_transformer_forward_backward_on_cuda():
    seed_everything(42, 1)
    config = next(item for item in deterministic_configurations()
                  if item["architecture"] == "transformer")
    model = DualBranchLongRiskModel(6, 4, config).to("cuda")
    hourly = torch.zeros((2, 168, 6), device="cuda")
    five = torch.zeros((2, 288, 4), device="cuda")
    loss = model(hourly, five).square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters()
               if parameter.requires_grad)


def test_robust_scaler_is_train_only_and_clipped():
    hourly = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    five = np.arange(4 * 5 * 2, dtype=np.float32).reshape(4, 5, 2)
    scaler = RobustSequenceScaler.fit(hourly[:3], five[:3])
    transformed_hourly, transformed_five = scaler.transform(hourly * 100, five * 100)
    assert np.max(transformed_hourly) <= 10
    assert np.max(transformed_five) <= 10
    np.testing.assert_array_equal(scaler.median_hourly, np.median(hourly[:3].reshape(-1, 2), axis=0))


def test_sequence_cache_boundaries_when_prepared():
    if not (OUT / "sequence_data_quality.csv").exists():
        pytest.skip("prepare stage has not run")
    import pandas as pd
    quality = pd.read_csv(OUT / "sequence_data_quality.csv")
    assert len(quality) == 2
    assert quality.rows.gt(0).all()
    if "all_execute_next_5m" in quality:
        assert quality.all_execute_next_5m.fillna(True).all()
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        with np.load(OUT / "sequence_cache" / f"{pair}_dual_sequence.npz") as values:
            assert values["hourly"].shape[1:] == (168, 26)
            assert values["five"].shape[1:] == (288, 10)
            assert np.all(values["sequence_end_ts"] + 300 == values["signal_ts"])


def test_final_contract_is_disabled_and_has_no_sell_when_available():
    path = OUT / "grid_hybrid_risk_gate_v1_sample.json"
    if not path.exists():
        pytest.skip("finalize/plot stages have not run")
    signal = json.loads(path.read_text(encoding="utf-8"))
    assert signal["schema"] == "grid-hybrid-risk-gate-v1"
    assert signal["deployment_allowed"] is False
    assert signal["market_sell_action"] is False
    assert signal["stop_excess_inventory"] is False
    assert signal["mechanism1_fallback_allowed"] is False
    assert all(not row["buy_enabled"] for row in signal["pairs"].values())
