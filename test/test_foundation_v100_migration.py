"""No-CUDA tests of the separate, forecast-only V100 migration entry points."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import prepare_foundation_v100_bundle as prep
from scripts import run_foundation_v100_forecasts as run


def contracts():
    cpu = {"context_days": 128, "local_fit": False, "test": ["start", "end"],
           "costs": {"base": {"fee": .001, "slippage": .0002}},
           "forecast_to_weights": {"hurdle": .0024, "cap": .25},
           "models": {"kronos_small": {"device": "cpu", "sampling_seeds": [0, 1, 2, 3, 4],
                                      "vendor_path": "old", "weights_path": "old", "context": 128}},
           "pinned_files": [{"path": "old", "sha256": "unchanged"}], "model_workers": 4}
    gpu = deepcopy(cpu)
    gpu["models"]["kronos_small"].update(device="cuda:0", vendor_path="/new/vendor", weights_path="/new/weights")
    gpu["model_workers"] = 2
    gpu["pinned_files"][0]["path"] = "/new/pinned"
    return cpu, gpu


def test_migration_preserves_financial_rule_and_sampling_contract():
    cpu, gpu = contracts()
    run.validate_contract(cpu, gpu)
    changed = deepcopy(gpu)
    changed["forecast_to_weights"]["hurdle"] = .001
    with pytest.raises(AssertionError, match="financial"):
        run.validate_contract(cpu, changed)
    changed = deepcopy(gpu)
    changed["models"]["kronos_small"]["sampling_seeds"] = [0]
    with pytest.raises(AssertionError, match="native model rule"):
        run.validate_contract(cpu, changed)
    changed = deepcopy(gpu)
    changed["pinned_files"][0]["sha256"] = "different"
    with pytest.raises(AssertionError, match="bytes"):
        run.validate_contract(cpu, changed)


@pytest.mark.parametrize("path", ["/", "relative", "C:/research", "/research/../other"])
def test_remote_destination_is_new_absolute_linux_path(path):
    with pytest.raises(ValueError):
        prep.validate_remote_root(path)


def test_two_physical_v100_uuids_are_probed_without_torch(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    raw = "0, Tesla V100-SXM2-32GB, GPU-one, 32768, 30000, 0, 550.00\n1, Tesla V100-SXM2-32GB, GPU-two, 32768, 29000, 5, 550.00\n"
    monkeypatch.setattr(run.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=raw))
    devices = run.gpu_probe([0, 1])
    assert [item["uuid"] for item in devices] == ["GPU-one", "GPU-two"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-one")
    with pytest.raises(RuntimeError, match="allocation"):
        run.gpu_probe([0, 1])


def test_worker_isolates_uuid_and_sets_cublas_before_cuda_initialization(monkeypatch):
    import os
    calls = []
    def set_device(index):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-two"
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        calls.append(index)
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           set_device=set_device, current_device=lambda: 0,
                           get_device_name=lambda index: "Tesla V100-SXM2-32GB",
                           get_device_capability=lambda index: (7, 0))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(run, "WORKER", None)
    for name in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM"):
        monkeypatch.delenv(name, raising=False)
    run.initialize_worker({"physical_index": 1, "uuid": "GPU-two", "name": "V100"})
    run.assert_worker()
    assert calls == [0] and run.WORKER["physical_index"] == 1 and run.WORKER["logical_device"] == "cuda:0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-one")
    with pytest.raises(AssertionError, match="binding"):
        run.assert_worker()


def test_bundle_excludes_cpu_forecasts_and_weights_and_preserves_originals(tmp_path):
    repo, source, output = tmp_path / "repo", tmp_path / "cpu", tmp_path / "bundle"
    repo.mkdir()
    source.mkdir()
    for entry in (*prep.ENTRYPOINTS, "scripts/financial.py"):
        path = repo / entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source preserved\n")
    vendors, sources = {}, []
    for key in ("kronos", "chronos2"):
        vendor = tmp_path / (key + "-commit")
        vendor.mkdir()
        file = vendor / "LICENSE"
        file.write_text("license")
        vendors[key] = vendor
        sources.append({"id": key, "vendor_path": str(vendor), "files": [
            {"path": "LICENSE", "local_path": str(file), "sha256": prep.digest(file)}]})
    models, pin = [], [{"path": item["files"][0]["local_path"], "sha256": item["files"][0]["sha256"]} for item in sources]
    configs = {}
    for key in ("kronos_small", "kronos_base", "chronos2", "kronos_tokenizer"):
        directory = tmp_path / key
        directory.mkdir()
        file = directory / "config.json"
        file.write_text("{}")
        models.append({"name": key, "local_directory": str(directory), "files": [
            {"path": "config.json", "local_path": str(file), "sha256": prep.digest(file)}]})
        pin.append({"path": str(file), "sha256": prep.digest(file)})
        if key != "kronos_tokenizer":
            configs[key] = {"device": "cpu", "vendor_path": str(vendors["chronos2" if key == "chronos2" else "kronos"]),
                            "weights_path": str(directory), "sampling_seeds": [] if key == "chronos2" else [0, 1, 2, 3, 4]}
    market = source / "market/daily_ohlcva.pkl.gz"
    market.parent.mkdir()
    market.write_bytes(b"prepared market bytes")
    prep.write(source / "market/manifest.json", {"market_sha256": prep.digest(market)})
    spec = {"local_fit": False, "context_days": 128, "models": configs, "pinned_files": pin,
            "market_manifest_sha256": prep.digest(source / "market/manifest.json"),
            "forecast_to_weights": {"hurdle": .0024}}
    prep.write(source / "frozen_spec.json", spec)
    prep.write(source / "inference_frozen_inputs.json", {"protocol_sha256": prep.digest(source / "frozen_spec.json"),
               "source_sha256": {"scripts/financial.py": prep.digest(repo / "scripts/financial.py")}})
    prep.write(source / "sources/source_manifest.json", {"sources": sources})
    prep.write(source / "sources/weights_manifest.json", {"models": models})
    for name in ("policy_precommit.json", "model_provenance.json", "model_provenance_evaluation.json"):
        prep.write(source / name, {})
    (source / "forecast_request_plan.csv").write_text("unchanged causal plan\n")
    (source / "forecasts").mkdir()
    (source / "forecasts/secret_cpu_prediction.csv").write_text("must not copy")
    original = {str(file): prep.digest(file) for file in source.rglob("*") if file.is_file()}
    manifest = prep.create_bundle(source, output, "/home/ubuntu/research/new_gpu", repository=repo)
    assert manifest["status"] == "prepared_not_executed"
    assert not any("secret_cpu_prediction" in entry["relative_path"] or entry["relative_path"].startswith("weights/") for entry in manifest["files"])
    template = prep.read(output / "experiment/gpu_spec_template.json")
    assert all(item["device"] == "cuda:0" for item in template["models"].values())
    assert template["forecast_to_weights"] == spec["forecast_to_weights"]
    assert len(prep.read(output / "weights_transfer_manifest.json")["files"]) == 4
    assert original == {str(file): prep.digest(file) for file in source.rglob("*") if file.is_file()}
