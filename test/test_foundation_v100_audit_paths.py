"""Failure-path checks for returned GPU proof aliases, without running models."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from scripts.foundation_model_proof_paths import ProofPaths, sha256_file
from scripts.foundation_model_audit_helpers import Checks
from scripts.audit_crypto_foundation_models import audit_gpu_backend, same_csv_records


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def proof_fixture(tmp_path):
    output = tmp_path / "returned"
    output.mkdir()
    originals = {}
    for name in ("weight", "provenance", "backend"):
        path = tmp_path / (name + ".proof")
        path.write_text("unaltered synthetic " + name, encoding="utf-8")
        originals[name] = path
    remote = {name: "/remote/experiment/" + name + ".proof" for name in originals}
    spec = {"pinned_files": [{"path": remote["weight"], "sha256": sha256_file(originals["weight"])}],
            "model_provenance_path": remote["provenance"], "model_provenance_sha256": sha256_file(originals["provenance"]),
            "backend_evidence_path": remote["backend"], "backend_evidence_sha256": sha256_file(originals["backend"])}
    write_json(output / "frozen_spec.json", spec)
    proof = output / "evidence/cpu/proof.json"
    write_json(proof, {"synthetic_only": True})
    imported = {"gpu_protocol_sha256": sha256_file(output / "frozen_spec.json"),
                "gpu_original_spec_and_contracts_unchanged": True, "prediction_values_changed": False,
                "cpu_gpu_prediction_equality_claimed": False,
                "proof_path_overrides": {remote[name]: str(path) for name, path in originals.items()},
                "files": [{"path": "frozen_spec.json", "sha256": sha256_file(output / "frozen_spec.json")},
                          {"path": "evidence/cpu/proof.json", "sha256": sha256_file(proof)}]}
    return output, spec, imported, originals, remote


def test_gpu_proof_aliases_resolve_exact_bytes_without_rewriting_original_spec(tmp_path):
    output, spec, imported, originals, remote = proof_fixture(tmp_path)
    before = (output / "frozen_spec.json").read_bytes()
    proof = ProofPaths(output, spec, imported)
    for name in originals:
        assert proof.resolve(remote[name]) == originals[name].resolve()
    assert proof.local_evidence("evidence/cpu/proof.json").is_file()
    assert (output / "frozen_spec.json").read_bytes() == before


def test_gpu_proof_rejects_unanchored_alias_substitution_and_changed_bytes(tmp_path):
    output, spec, imported, originals, remote = proof_fixture(tmp_path)
    changed = deepcopy(imported)
    changed["proof_path_overrides"]["/unanchored/source"] = str(originals["weight"])
    with pytest.raises(AssertionError, match="correspond exactly"):
        ProofPaths(output, spec, changed)
    changed = deepcopy(imported)
    changed["proof_path_overrides"][remote["weight"]] = str(originals["backend"])
    with pytest.raises(AssertionError, match="frozen SHA256"):
        ProofPaths(output, spec, changed)
    proof = ProofPaths(output, spec, imported)
    originals["weight"].write_text("tampered", encoding="utf-8")
    with pytest.raises(AssertionError, match="Proof bytes changed"):
        proof.resolve(remote["weight"])


def test_gpu_local_evidence_rejects_parent_traversal_unlisted_or_changed_copy(tmp_path):
    output, spec, imported, originals, remote = proof_fixture(tmp_path)
    proof = ProofPaths(output, spec, imported)
    with pytest.raises(ValueError, match="stay inside"):
        proof.local_evidence("../weight.proof")
    unlisted = output / "unlisted.json"
    write_json(unlisted, {})
    with pytest.raises(AssertionError, match="imported SHA list"):
        proof.local_evidence("unlisted.json")
    write_json(output / "evidence/cpu/proof.json", {"tampered": True})
    with pytest.raises(AssertionError, match="imported SHA list"):
        proof.local_evidence("evidence/cpu/proof.json")


def test_gpu_proof_rejects_changed_protocol_or_imported_traversal(tmp_path):
    output, spec, imported, originals, remote = proof_fixture(tmp_path)
    changed = deepcopy(imported)
    changed["gpu_protocol_sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="unchanged GPU protocol"):
        ProofPaths(output, spec, changed)
    changed = deepcopy(imported)
    changed["files"].append({"path": "../unanchored.txt", "sha256": "0" * 64})
    with pytest.raises(ValueError, match="normalized relative"):
        ProofPaths(output, spec, changed)


def gpu_backend_fixture(tmp_path):
    keys = ("kronos_small", "kronos_base", "chronos2")
    cpu = {"test": ["2025-10-23", "2026-10-23"], "capital": 10000,
           "pinned_files": [{"path": "original", "sha256": "original-sha"}],
           "models": {key: {"device": "cpu", "first_decision_date": "2025-10-21", "sampling_seeds": []}
                      for key in keys}}
    cpu_path = tmp_path / "cpu.json"
    write_json(cpu_path, cpu)
    spec = deepcopy(cpu)
    spec["migration"] = {"parent_cpu_protocol_sha256": sha256_file(cpu_path)}
    for config in spec["models"].values():
        config["device"] = "cuda:0"
    control = {"parent_cpu_protocol_sha256": sha256_file(cpu_path), "gpu_indices": [0, 1],
               "expected_parameter_sha256_by_model": {key: "actual-" + key for key in keys}}
    control_path = tmp_path / "control.json"
    write_json(control_path, control)
    backend = {"gpu_probe": [{"physical_index": index, "uuid": "GPU-" + str(index), "name": "Tesla V100-SXM2-16GB"}
                             for index in (0, 1)],
               "no_cpu_fallback": True, "new_models_trained": 0, "financial_evaluation_performed": False,
               "cpu_gpu_forecast_bitwise_equality_claimed": False, "synthetic_checks": []}
    for index in (0, 1):
        uuid = "GPU-" + str(index)
        backend["synthetic_checks"].append({"binding": {"uuid": uuid, "visible_devices": uuid,
              "torch_current_device": 0, "logical_device": "cuda:0", "compute_capability": [7, 0]},
              "models": [{"model_key": key, "status": "passed", "synthetic_only": True,
                          "same_gpu_repeat_equal": True, "input_unchanged": True, "batch": 10,
                          "context_days": 128, "horizon_days": 1, "parameter_sha256": "actual-" + key}
                         for key in keys]})

    class SyntheticPaths:
        def local_evidence(self, name):
            return cpu_path if name == "evidence/cpu/frozen_spec.json" else control_path
    return spec, backend, SyntheticPaths()


def test_gpu_backend_accepts_two_isolated_devices_and_real_parameter_repeat_records(tmp_path):
    spec, backend, paths = gpu_backend_fixture(tmp_path)
    review = Checks()
    audit_gpu_backend(review, tmp_path, spec, backend, paths)
    assert not review.failures


@pytest.mark.parametrize("mutation", ("same_uuid", "wrong_binding", "parameter_substitution", "financial_change", "cpu_fallback"))
def test_gpu_backend_rejects_false_hardware_or_native_evidence(tmp_path, mutation):
    spec, backend, paths = gpu_backend_fixture(tmp_path)
    if mutation == "same_uuid":
        backend["gpu_probe"][1]["uuid"] = backend["gpu_probe"][0]["uuid"]
    elif mutation == "wrong_binding":
        backend["synthetic_checks"][0]["binding"]["visible_devices"] = "GPU-other"
    elif mutation == "parameter_substitution":
        backend["synthetic_checks"][0]["models"][0]["parameter_sha256"] = "changed"
    elif mutation == "financial_change":
        spec["capital"] = 20000
    elif mutation == "cpu_fallback":
        backend["no_cpu_fallback"] = False
    review = Checks()
    audit_gpu_backend(review, tmp_path, spec, backend, paths)
    assert review.failures


def test_csv_comparison_accepts_storage_dtype_only_with_exact_python_scalar_values():
    daily = pd.DataFrame({"asset_key": ["1", "1027"], "predicted_amount": pd.Series([123, 456], dtype="int64"),
                          "forecast_reason": [np.nan, np.nan], "valid_forecast": [True, True]})
    merged = daily.copy()
    merged["predicted_amount"] = merged.predicted_amount.astype("float64")
    assert not daily.equals(merged) and same_csv_records(daily, merged)


@pytest.mark.parametrize("left,right", [
    (9007199254740993, 9007199254740992.0),
    (1., np.nextafter(1., 2.)),
    (np.nan, 0.),
    ("1", 1),
    (True, 1),
])
def test_csv_comparison_rejects_any_real_value_change_without_float_coercion(left, right):
    assert not same_csv_records(pd.DataFrame({"value": [left]}), pd.DataFrame({"value": [right]}))


def test_csv_comparison_rejects_changed_columns_or_missing_rows():
    assert not same_csv_records(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))
    assert not same_csv_records(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1, 2]}))
