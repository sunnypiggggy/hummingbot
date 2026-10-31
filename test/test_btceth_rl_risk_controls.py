"""Holdout controls remain closed without pre-test contracts; no real test prices."""
from datetime import timedelta
import json
from pathlib import Path

import pytest

from scripts import btceth_rl_risk_controls as risk


def put(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def prepared(tmp_path):
    reference, main, source = (tmp_path / name for name in ("reference", "main", "source"))
    for path in (reference, main, source / "scripts", tmp_path / "data"):
        path.mkdir(parents=True)
    original = risk.REFERENCE
    for name in ("run_risk_budget_controls.py", "independent_audit_source_snapshot.py", "protocol.json",
                 "completion.json", "serialized_verification.json"):
        (reference / name).write_bytes((original / name).read_bytes())
    (reference / "summary.csv").write_text("synthetic validation summary pin\n")
    (reference / "monthly_returns.csv").write_text("synthetic validation monthly pin\n")
    code = {}
    for name in risk.FINANCIAL:
        dest = source / name
        dest.write_bytes((risk.ROOT / name).read_bytes())
        code[name] = risk.sha(dest)
    dataset = tmp_path / "data/dataset.pkl.gz"
    dataset.write_bytes(b"Deliberately not a pickle: prepare must only hash it.")
    (dataset.parent / "manifest.json").write_text("{}\n")
    spec = {"train": list(risk.TRAIN), "validation": list(risk.VALIDATION), "test": list(risk.TEST),
            "costs": risk.COSTS, "code_sha256": code, "dataset_sha256": risk.sha(dataset),
            "dataset_manifest_sha256": risk.sha(dataset.parent / "manifest.json")}
    put(main / "frozen_spec.json", spec)
    risk.prepare(reference, main, dataset, source_root=source)
    return reference, main, dataset, source


def ready(prepared):
    reference, main, dataset, source = prepared
    p = risk.read_json(reference / "test_protocol.json")
    started = risk.stamp(p["prepared_at"]) + timedelta(seconds=10)
    original = {"frozen_at": (started - timedelta(seconds=5)).isoformat(),
                "primary_validation_leader": {"algorithm": "sac", "profile": "balanced"}}
    put(main / "validation_selection.json", original)
    marker = {"at": started.isoformat(), "validation_selection_sha256": risk.sha(main / "validation_selection.json"),
              "frozen_spec_sha256": risk.sha(main / "frozen_spec.json")}
    put(main / "test_started.json", marker)
    for name in ("goal_candidate_protocol.json", "goal_candidate_selection.json", "goal_candidate_selection_receipt.json"):
        put(main / name, {})
    return {"test_pnl_read": False, "test_started": marker, "original_selection": original,
            "new_model_validation_leader": {"algorithm": "recurrent_ppo", "profile": "balanced"}}


def test_prepare_only_hashes_unpicklable_market_and_preserves_byte_sources(prepared):
    reference, main, dataset, source = prepared
    p = risk.verify_protocol(reference, dataset, source_root=source)
    assert p["dataset_sha256"] == risk.sha(dataset)
    assert not (reference / "test").exists()
    for name in ("run_risk_budget_controls.py", "independent_audit_source_snapshot.py"):
        assert (reference / name).read_bytes() == (reference / "holdout_sources" / name).read_bytes()
    assert risk.read_json(reference / "test_protocol_receipt.json")["protocol_sha256"] == risk.sha(reference / "test_protocol.json")


def test_prepare_refuses_to_overwrite_protocol_or_snapshots(prepared):
    reference, main, dataset, source = prepared
    before = risk.sha(reference / "test_protocol.json")
    with pytest.raises(FileExistsError):
        risk.prepare(reference, main, dataset, source_root=source)
    assert risk.sha(reference / "test_protocol.json") == before


def test_missing_contracts_refuse_before_loading_or_creating_test(prepared, monkeypatch):
    reference, main, dataset, source = prepared
    monkeypatch.setattr(risk, "load_core", lambda *a, **k: pytest.fail("No formula or data loading before gate"))
    with pytest.raises(risk.GateClosed, match="missing sealed main contracts"):
        risk.run_test(reference, main, dataset, source_root=source)
    assert not (reference / "test").exists()


def test_gate_delegates_to_strict_read_only_goal_validator(prepared, monkeypatch):
    reference, main, dataset, source = prepared
    context = ready(prepared)
    calls = []
    monkeypatch.setattr(risk, "goal_validator", lambda root: calls.append(root) or context)
    value = risk.validate_test_gate(reference, main, dataset, source_root=source)
    assert calls == [main] and value["passed"] and value["test_pnl_read"] is False
    assert value["new_model_validation_leader"]["algorithm"] != "sac"
    assert not (reference / "test").exists()


def test_own_contract_tampering_is_rejected_by_whole_file_receipt(prepared, monkeypatch):
    reference, main, dataset, source = prepared
    context = ready(prepared)
    monkeypatch.setattr(risk, "goal_validator", lambda root: context)
    protocol = risk.read_json(reference / "test_protocol.json")
    protocol["volatility_rule"]["target_annual_vol"] = .21
    put(reference / "test_protocol.json", protocol)
    with pytest.raises(risk.GateClosed, match="receipt"):
        risk.validate_test_gate(reference, main, dataset, source_root=source)


def test_late_risk_receipt_cannot_backfill_an_already_started_test(prepared, monkeypatch):
    reference, main, dataset, source = prepared
    context = ready(prepared)
    monkeypatch.setattr(risk, "goal_validator", lambda root: context)
    receipt = risk.read_json(reference / "test_protocol_receipt.json")
    receipt["sealed_at"] = (risk.stamp(context["test_started"]["at"]) + timedelta(seconds=1)).isoformat()
    put(reference / "test_protocol_receipt.json", receipt)
    with pytest.raises(risk.GateClosed, match="before main test dispatch"):
        risk.validate_test_gate(reference, main, dataset, source_root=source)


def test_goal_hash_mismatch_stops_before_any_output_or_market_loading(prepared, monkeypatch):
    reference, main, dataset, source = prepared
    ready(prepared)
    def reject(root):
        raise ValueError("Goal candidate selection receipt SHA mismatch")
    monkeypatch.setattr(risk, "goal_validator", reject)
    monkeypatch.setattr(risk, "load_core", lambda *a, **k: pytest.fail("Rejected gate must not load data"))
    with pytest.raises(ValueError, match="SHA mismatch"):
        risk.run_test(reference, main, dataset, source_root=source)
    assert not (reference / "test").exists()


def test_changed_financial_source_is_rejected(prepared, monkeypatch):
    reference, main, dataset, source = prepared
    context = ready(prepared)
    monkeypatch.setattr(risk, "goal_validator", lambda root: context)
    with (source / risk.FINANCIAL[0]).open("a") as stream:
        stream.write("\n# altered source\n")
    with pytest.raises(risk.GateClosed, match="financial source"):
        risk.validate_test_gate(reference, main, dataset, source_root=source)


def test_synthetic_formula_and_future_causality_without_market_unpickling(prepared, monkeypatch):
    reference, main, dataset, source = prepared
    from scripts import backtest_btceth_rl
    monkeypatch.setattr(backtest_btceth_rl, "load_dataset", lambda *a, **k: pytest.fail("Synthetic check cannot read market pickle"))
    result = risk.formula_sample_check(reference, output=reference / "synthetic", source_root=source, dataset=dataset)
    assert result["synthetic_only"] and result["original_dataset_loaded"] is False
    assert result["hand_calculation_passed"] and result["future_truncation_exact"] and result["future_mutation_exact"]
