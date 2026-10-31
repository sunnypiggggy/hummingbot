"""Boolean-to-marker shape adaptation only; no original annual prices."""
from pathlib import Path
import json

import pytest

from scripts import btceth_rl_risk_controls_adapter as a


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def main(tmp_path):
    main = tmp_path / "main"
    write(main / "frozen_spec.json", {"synthetic": True})
    write(main / "validation_selection.json", {"synthetic": True})
    write(main / "test_started.json", dict(at="2026-10-29T19:05:00.445842+00:00",
        frozen_spec_sha256=a.sha(main / "frozen_spec.json"), validation_selection_sha256=a.sha(main / "validation_selection.json")))
    return main


def test_original_validator_runs_first_context_copied_and_shape_only_changed(main):
    original_context = {"test_started": True, "test_pnl_read": False, "other": object()}
    calls = []
    def original(path):
        calls.append(path)
        return original_context
    got = a.adapted_validator(original, a.sha(main / "test_started.json"))(main)
    assert calls == [main] and original_context["test_started"] is True
    assert got["test_started"]["at"] == "2026-10-29T19:05:00.445842+00:00"
    assert got["other"] is original_context["other"] and got["test_pnl_read"] is False


@pytest.mark.parametrize("context", [{"test_started": False, "test_pnl_read": False},
    {"test_started": True, "test_pnl_read": True}, {"test_pnl_read": False}, {"test_started": 1, "test_pnl_read": False}])
def test_false_missing_or_pnl_read_context_cannot_be_adapted(main, context):
    with pytest.raises(ValueError, match="Original validator"):
        a.adapted_validator(lambda p: context, a.sha(main / "test_started.json"))(main)


def test_wrong_marker_sha_fails_before_original_validator_or_data(main):
    with pytest.raises(ValueError, match="before original"):
        a.adapted_validator(lambda p: pytest.fail("No validator after wrong marker SHA"), "0" * 64)(main)


def test_missing_marker_is_not_fabricated_from_context(main):
    (main / "test_started.json").unlink()
    with pytest.raises(FileNotFoundError):
        a.adapted_validator(lambda p: pytest.fail("No validator before actual marker"), "0" * 64)(main)


def test_spec_or_selection_sha_and_mid_validation_marker_mutation_rejected(main):
    expected = a.sha(main / "test_started.json")
    def mutate(path):
        write(main / "test_started.json", {"at": "2026-10-30T00:00:00Z"})
        return {"test_started": True, "test_pnl_read": False}
    with pytest.raises(ValueError, match="marker/selection/spec"):
        a.adapted_validator(mutate, expected)(main)


@pytest.mark.parametrize("throws", [False, True])
def test_original_function_restored_on_success_and_failure(main, monkeypatch, throws):
    original = lambda p: {"test_started": True, "test_pnl_read": False}
    monkeypatch.setattr(a.risk, "goal_validator", original)
    try:
        with a.compatible_validator(a.sha(main / "test_started.json")):
            assert a.risk.goal_validator is not original
            assert isinstance(a.risk.goal_validator(main)["test_started"], dict)
            if throws: raise RuntimeError("synthetic")
    except RuntimeError:
        assert throws
    assert a.risk.goal_validator is original


def test_existing_test_or_adapter_cannot_redeclare_or_rerun(tmp_path):
    ref, attempt = tmp_path / "reference", tmp_path / "reference/adapter_attempts/compat_20261030"
    (ref / "test").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        a.prepare(ref, tmp_path / "main", tmp_path / "data", attempt, tmp_path / "failed")


def test_frozen_original_implementations_stay_byte_identical():
    values = a.source_pins()
    assert values["scripts/btceth_rl_risk_controls.py"] == a.RISK_SHA
    assert values["scripts/btceth_rl_goal_candidate.py"] == a.GOAL_SHA


def test_original_risk_run_gate_still_blocks_before_core_or_price_loading(main, tmp_path, monkeypatch):
    ref = tmp_path / "reference"; ref.mkdir()
    for name in a.risk.REQUIRED_GATE:
        if not (main / name).exists(): write(main / name, {})
    monkeypatch.setattr(a.risk, "verify_protocol", lambda *args, **kwargs: {
        "original_frozen_spec_sha256": a.sha(main / "frozen_spec.json"), "prepared_at": "2026-10-29T16:00:00Z"})
    monkeypatch.setattr(a.risk, "goal_validator", lambda p: {"test_started": False, "test_pnl_read": False})
    monkeypatch.setattr(a.risk, "load_core", lambda *args, **kwargs: pytest.fail("No source/core/price loading after closed original gate"))
    with a.compatible_validator(a.sha(main / "test_started.json")):
        with pytest.raises(ValueError, match="Original validator"):
            a.risk.run_test(ref, main, tmp_path / "NEVER_OPEN_THIS_PICKLE")
    assert not (ref / "test").exists()
