"""Synthetic artifact tests; never execute native models or real-price ledgers."""
from concurrent.futures import Future
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import import_foundation_v100_forecasts as importer


def test_remote_absolute_paths_are_mapped_only_inside_new_local_root(tmp_path):
    target = importer.map_remote_path("/home/sunny/new/experiment/forecasts/model/seed_0.csv",
                                      "/home/sunny/new/experiment", tmp_path / "local")
    assert target == tmp_path / "local/forecasts/model/seed_0.csv"
    for value in ("/other/seed.csv", "/home/sunny/new/experiment/../other.csv", "C:/old.csv", "relative.csv"):
        with pytest.raises(ValueError):
            importer.map_remote_path(value, "/home/sunny/new/experiment", tmp_path / "local")


def test_derived_model_manifest_preserves_numbers_and_original_object(tmp_path):
    original = {"parameter_sha256_before": "same", "parameter_sha256_after": "same",
                "forecast_files": [{"path": "/remote/experiment/forecasts/model/seed_0.csv",
                                    "seed": 0, "sha256": "prediction_bytes", "rows": 123}],
                "config": {"device": "cuda:0", "weights_path": "/remote/weights"},
                "native_repeat_verifications": [{"date": "synthetic", "passed": True}]}
    saved = deepcopy(original)
    derived = importer.derive_model_manifest(original, original_sha256="original_manifest_hash",
                                             remote_experiment="/remote/experiment", local_experiment=tmp_path)
    assert original == saved
    assert derived["forecast_files"][0]["path"] == str(tmp_path / "forecasts/model/seed_0.csv")
    assert derived["forecast_files"][0]["sha256"] == "prediction_bytes"
    assert derived["config"] == original["config"]
    assert derived["artifact_relocation"]["prediction_values_changed"] is False
    assert derived["artifact_relocation"]["gpu_protocol_and_cache_contracts_changed"] is False


def test_receipt_rejects_different_returned_bundle_before_model_or_dataset_access(tmp_path):
    prepared, returned = tmp_path / "prepared", tmp_path / "returned"
    prepared.mkdir()
    returned.mkdir()
    importer.write(prepared / "migration_control.json", {"trusted": True})
    importer.write(returned / "migration_control.json", {"trusted": False})
    with pytest.raises(AssertionError, match="locally prepared bundle"):
        importer.verify_gpu_receipt(returned, tmp_path / "cpu", prepared)


def test_import_integrity_detects_changed_prediction_bytes_without_gpu(monkeypatch, tmp_path):
    importer.write(tmp_path / "frozen_spec.json", {"synthetic_test_fixture": True})
    importer.write(tmp_path / "import_cache_audit.json", {"status": "passed"})
    prediction = tmp_path / "forecasts/synthetic.csv"
    prediction.parent.mkdir()
    prediction.write_bytes(b"synthetic original prediction bytes")
    importer.write(tmp_path / "local_import_manifest.json", {
        "gpu_protocol_sha256": importer.digest(tmp_path / "frozen_spec.json"),
        "files": [{"path": "forecasts/synthetic.csv", "sha256": importer.digest(prediction)}]})
    monkeypatch.setattr(importer, "assert_frozen_local_sources", lambda *args: None)
    importer.verify_imported(tmp_path)
    prediction.write_bytes(b"changed")
    with pytest.raises(AssertionError, match="changed"):
        importer.verify_imported(tmp_path)


def test_cpu_report_evidence_is_copied_byte_for_byte_and_existing_alias_is_unchanged(tmp_path):
    source, target = tmp_path / "cpu", tmp_path / "new_gpu_import"
    originals = {}
    for index, name in enumerate((*importer.CPU_SOURCE_EVIDENCE, *importer.PUBLIC_SOURCE_EVIDENCE)):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"original-evidence-{index}\r\n".encode())
        originals[name] = path.read_bytes()
    alias = target / "policy_precommit.json"
    alias.parent.mkdir(parents=True)
    alias.write_bytes(originals["policy_precommit.json"])
    records = importer.copy_cpu_source_evidence(source, target)
    assert len(records) == len(originals)
    for row in records:
        name = row["local_relative_path"]
        assert (target / name).read_bytes() == originals[name] == (source / name).read_bytes()
        assert row["local_sha256"] == row["source_sha256"] == importer.digest(source / name)
        assert row["classification"] == "original_cpu_source_evidence_not_gpu_runtime"


def test_cpu_evidence_alias_conflict_is_rejected_without_overwriting_either_side(tmp_path):
    source, target = tmp_path / "cpu", tmp_path / "new_gpu_import"
    for name in importer.CPU_SOURCE_EVIDENCE:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original\r\n")
    alias = target / "original_research_preservation.json"
    alias.parent.mkdir(parents=True)
    alias.write_bytes(b"different\n")
    with pytest.raises(AssertionError, match="differs"):
        importer.copy_cpu_source_evidence(source, target)
    assert alias.read_bytes() == b"different\n"
    assert (source / "original_research_preservation.json").read_bytes() == b"original\r\n"


class InlineExecutor:
    def __init__(self, **kwargs):
        assert kwargs == {"max_workers": 4}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, task):
        future = Future()
        future.set_result(function(task))
        return future


def test_evaluate_dispatches_existing_case_task_for_six_cases_and200_repeats(monkeypatch, tmp_path):
    # Fake results exist only in this temporary synthetic fixture directory.
    from scripts import backtest_crypto_foundation_models as financial
    spec = {"models": {key: {"synthetic_test_fixture": True} for key in ("kronos_small", "kronos_base", "chronos2")}}
    imported = {"gpu_protocol_sha256": "synthetic_gpu_protocol"}
    monkeypatch.setattr(importer, "verify_imported", lambda path: (imported, spec))
    monkeypatch.setattr(importer, "ProcessPoolExecutor", InlineExecutor)
    calls = []
    def fake_case(task):
        calls.append(task)
        return {"variant": task["variant"], "model_key": task["model_key"],
                "runs": 20 if task["model_key"] == "chronos2" else 40,
                "manifest_sha256": "synthetic_only"}
    monkeypatch.setattr(financial, "case_task", fake_case)
    monkeypatch.setattr(financial, "diagnostics", lambda *args: None)
    importer.write(tmp_path / "local_import_manifest.json", {"synthetic_test_fixture": True})
    importer.evaluate_imported(tmp_path)
    result = importer.read(tmp_path / "numeric_completion.json")
    assert len(calls) == 6 and result["evaluations"] == 200 and result["repeat_checks"] == 200
    assert {(task["variant"], task["model_key"]) for task in calls} == {
        (variant, key) for variant in ("original", "ex_ada") for key in spec["models"]}
    assert result["new_models_trained"] == 0
