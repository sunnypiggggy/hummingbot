"""Pure source/receipt/metadata checks; no market, NN, GPU, SSH or training."""
from pathlib import Path
import json
import pandas as pd
import pytest

from scripts import audit_btceth_rl_droq as bf
from scripts import btceth_rl_droq_parallel_audit as adapter


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path):
    output, main = tmp_path / "study", tmp_path / "main"
    output.mkdir(); main.mkdir()
    root = Path(__file__).resolve().parents[1]
    obj = bf.Inputs(output, main, root, bf.Checks())
    return obj


def test_training_time_view_has_no_fake_success_or_disk_mutation(inputs):
    path = inputs.roots["study"] / "fit_process.json"
    put(path, dict(status="failed", exit_code=1, started_at="2026-10-30T00:00:00Z", completed_at="2026-10-30T01:00:00Z"))
    original = path.read_bytes()
    envelope = dict(kind="derived_training_time_bounds_only", started_at="2026-10-30T00:00:00Z",
        completed_at="2026-10-30T02:00:00Z", model_count=5, original_terminal_relabelled=False,
        original_terminal_status="failed", accelerated_terminal_status="completed")
    view = adapter.TrainingTimeBounds(inputs, envelope)
    actual = view.json("study", "fit_process.json")
    assert set(actual) == {"kind", "started_at", "completed_at"}
    assert path.read_bytes() == original
    assert inputs.hashes["study"]["fit_process.json"] == adapter.sha(path)
    for key, value in (("model_count", 4), ("original_terminal_relabelled", True), ("original_terminal_status", "completed")):
        with pytest.raises(ValueError):
            adapter.TrainingTimeBounds(inputs, dict(envelope, **{key: value}))


@pytest.fixture
def receipts(inputs):
    study = inputs.roots["study"]
    dataset = study / "explicitly_not_a_dataset"; dataset.write_bytes(b"Synthetic hash only")
    source = {"scripts/audit_btceth_rl_droq.py": "synthetic_bf33_pin"}
    put(study / "protocol.json", dict(source_sha256=source))
    root = study / "cpu_receipts_v2"; root.mkdir()
    rows = []
    for index in range(24):
        run_id = f"synthetic_{index}"
        case = study / "validation" / run_id; case.mkdir(parents=True)
        for name in adapter.CASE_FILES:
            (case / name).write_bytes(b"Synthetic case bytes")
        folder = root / run_id; folder.mkdir()
        put(folder / "run.json", dict(run_id=run_id, reconstructed=True))
        (folder / "history.csv").write_text("timestamp,equity,net_return,drawdown\n2025-01-01,10000,0,0\n")
        put(folder / "monthly.json", {"2025-01": .0123, "2025-02": -.02})
        receipt = dict(scope="original_bf33_cpu_financial_case_audit", run_id=run_id, split="validation", worker_pid=1000 + index,
            wrapper_source_sha256=adapter.PINS["btceth_rl_droq_case_audit_v2.py"], dataset_sha256=adapter.sha(dataset),
            source_sha256=source, directory_relative=case.relative_to(study).as_posix(),
            case_sha256={name: adapter.sha(case / name) for name in adapter.CASE_FILES},
            result_sha256={name: adapter.sha(folder / name) for name in ("run.json", "history.csv", "monthly.json")},
            passed=True, checks_total=10, failure_count=0, failures=[], maximum_errors={"actual_metric": 1e-12})
        put(folder / "receipt.json", receipt)
        rows.append(dict(run_id=run_id, passed=True, receipt_sha256=adapter.sha(folder / "receipt.json"), receipt_directory=str(folder)))
    put(root / "manifest.json", dict(scope="parallel_bf33_financial_cases_only_not_full_study_audit", phase="validation", case_count=24,
        source_sha256=adapter.PINS["btceth_rl_droq_case_audit_v2.py"], frozen_auditor_sha256=source["scripts/audit_btceth_rl_droq.py"],
        protocol_sha256=adapter.sha(study / "protocol.json"), dataset_sha256=adapter.sha(dataset), passed=True, records=rows))
    return root, dataset


def test_adopt_preserves_original_monthly_dictionary_and_real_check_counts(inputs, receipts):
    root, dataset = receipts
    expected = adapter.adopt_case_receipts(inputs, bf, root, dataset)
    assert len(expected) == 24 and inputs.checks.failed == 0 and inputs.checks.count >= 240
    assert expected["synthetic_0"][2] == {"2025-01": .0123, "2025-02": -.02}
    assert isinstance(expected["synthetic_0"][1], pd.DataFrame)


@pytest.mark.parametrize("change", ["case", "output", "manifest"])
def test_receipt_adoption_rejects_changed_financial_inputs_or_outputs(inputs, receipts, change):
    root, dataset = receipts
    if change == "case":
        (inputs.roots["study"] / "validation/synthetic_0/fills.csv").write_bytes(b"Changed fill")
    elif change == "output":
        put(root / "synthetic_0/monthly.json", {"2025-01": 999.})
    else:
        value = json.loads((root / "manifest.json").read_text()); value["records"] = value["records"][:-1]
        put(root / "manifest.json", value)
    with pytest.raises(ValueError):
        adapter.adopt_case_receipts(inputs, bf, root, dataset)


def test_validation_inventory_contains_true_parallel_evidence_not_legacy_terminals():
    names = adapter.validation_files()
    assert len(names) == 217 and "parallel_validation_process.json" in names
    assert "validate_process.json" not in names and "validate.log" not in names
    assert len([name for name in names if name.endswith("/replay.json")]) == 24


def test_metadata_counter_merge_rejects_boolean_or_negative_counts():
    for value in (dict(checks_total=True, failure_count=0), dict(checks_total=1, failure_count=-1)):
        with pytest.raises(ValueError):
            adapter.merge_checks(bf.Checks(), value)
