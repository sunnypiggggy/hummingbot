"""Synthetic/accounting tests only; no trained models, market files or GPU."""
from concurrent.futures import Future
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import btceth_rl_droq as original
from scripts import btceth_rl_droq_parallel_eval as parallel

pytestmark = pytest.mark.timeout(1200)


class ConstantSyntheticPolicy:
    def __init__(self, seed, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.tensor = torch.tensor([float(seed)], dtype=torch.float32)
        self.policy = self

    def state_dict(self):
        return {"explicit_synthetic_not_trained": self.tensor}

    def predict(self, obs, deterministic=True):
        assert deterministic and np.asarray(obs).shape == (27,)
        return self.action.copy(), None


@pytest.fixture(scope="module")
def synthetic_comparison(tmp_path_factory):
    assert not torch.cuda.is_initialized()
    root = tmp_path_factory.mktemp("droq_parallel_synthetic_only")
    main = root / "synthetic_reference"; main.mkdir()
    dataset = root / "not_a_market_pickle"; dataset.write_bytes(b"Never deserialize this fixture")
    serial, split = root / "original", root / "parallel"
    serial.mkdir(); split.mkdir()
    protocol = dict(explicitly_synthetic=True, no_trained_models=True, no_cuda=True)
    for output in (serial, split):
        original.write_new(output / "protocol.json", protocol)
    start, end = map(pd.Timestamp, original.VALIDATION)
    dates = pd.date_range(start - pd.Timedelta(days=210), end, freq="h", inclusive="left")
    k = np.arange(len(dates))
    values = pd.DataFrame({"BTCUSDT": 100 + .001 * k + 2 * np.sin(k / 170),
                           "ETHUSDT": 200 + .001 * k + 3 * np.sin(k / 83)}, index=dates)
    data = {"hourly_" + key: values.copy() for key in ("open", "high", "low", "close")}
    data["hourly_volume"] = values * 0 + 100
    spec = dict(family_gpu_assignments={"raw": "SYNTHETIC-GPU-0", "cap075": "SYNTHETIC-GPU-1"})
    models = {seed: ConstantSyntheticPolicy(seed, action) for seed, action in enumerate(
        [(-1., -1.), (1., -1.), (1., 1.), (.5, 0.), (0., .5)])}
    for output in (serial, split):
        for seed, model in models.items():
            original.write_new(output / "training" / original.job_id(seed) / "record.json",
                dict(algorithm="droq", profile="balanced", seed=seed,
                    explicitly_synthetic_not_trained=True, parameter_sha256=original.parameter_digest(model)))
    current = {}; loads = []

    def setup(_):
        return dict(visible_gpu_uuid=spec["family_gpu_assignments"][current["family"]],
            device="cuda:0", name="V100 mocked only; actual tensors CPU", capability=[7, 0], explicitly_mocked=True)

    def load_policy(directory, *, upstream, device):
        assert device == "cuda:0" and upstream.parent in (serial, split)
        seed = int(Path(directory).name.rsplit("seed", 1)[1]); loads.append(seed)
        return models[seed]

    def load_data(path):
        assert path == dataset
        return data

    results = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(original, "protocol", lambda *args: spec)
        patch.setattr(original.shared, "torch_setup", setup)
        patch.setattr(original.shared, "load_dataset", load_data)
        patch.setattr(original, "load_policy", load_policy)
        patch.setattr(parallel, "get_runner", lambda *args: original)
        before = original.rng_digest()
        for family in original.BUDGETS:
            current["family"] = family
            original.evaluate_worker(dict(main=str(main), dataset=str(dataset), output=str(serial),
                family=family, split="validation"))
        assert loads == list(range(5)) * 2
        loads.clear()
        for plan in parallel.plan_cases():
            current["family"] = plan["family"]
            results.append(parallel.case_worker(dict(plan, main=str(main), dataset=str(dataset), output=str(split),
                protocol_sha256=original.sha(split / "protocol.json"), source_sha256=parallel.sha(parallel.__file__))))
        assert loads == ([0, 1, 2, 3, 4] * 2 + list(range(5)) * 2) * 2
        # Pretend futures completed in exactly the opposite order.
        completion = parallel.merge_groups(split, "validation", list(reversed(results)))
        assert before == original.rng_digest()
    for name in original.GROUP_FILES[:3]:
        frames = [pd.read_csv(serial / "validation" / family / "balanced" / name,
                              float_precision="round_trip") for family in original.BUDGETS]
        (serial / "validation" / name).write_bytes(original.canonical_csv(pd.concat(frames, ignore_index=True)))
    assert not torch.cuda.is_initialized()
    yield SimpleNamespace(serial=serial, parallel=split, results=results, completion=completion, spec=spec)
    assert not torch.cuda.is_initialized()


def test_all_24_accounts_match_original_f146_saved_bytes(synthetic_comparison):
    flow = synthetic_comparison
    for plan in parallel.plan_cases():
        folder = Path("validation") / plan["family"] / "balanced" / plan["scenario"] / plan["key"]
        for name in original.CASE_FILES:
            assert (flow.serial / folder / name).read_bytes() == (flow.parallel / folder / name).read_bytes(), str(folder / name)
    for family in original.BUDGETS:
        for name in original.GROUP_FILES[:3]:
            folder = Path("validation") / family / "balanced" / name
            assert (flow.serial / folder).read_bytes() == (flow.parallel / folder).read_bytes()
        complete = original.read_json(flow.parallel / "validation" / family / "balanced/completion.json")
        assert complete["execution_mode"] == "parallel_cases" and complete["worker_pid"] is None
        assert len(complete["case_workers"]) == complete["case_count"] == complete["repeat_count"] == 12
        assert all(item["device"]["visible_gpu_uuid"] == flow.spec["family_gpu_assignments"][family]
                   for item in complete["case_workers"])


def test_reverse_completion_order_preserves_all_parent_csv_bytes(synthetic_comparison):
    flow = synthetic_comparison
    for name in original.GROUP_FILES[:3]:
        assert (flow.serial / "validation" / name).read_bytes() == (flow.parallel / "validation" / name).read_bytes()
    frames = [pd.read_csv(flow.parallel / "validation" / name, float_precision="round_trip") for name in original.GROUP_FILES[:3]]
    assert [len(frame) for frame in frames] == [24, 24 * 296, 24 * 10]
    assert flow.completion["case_count"] == flow.completion["repeat_count"] == 24
    assert len(flow.completion["case_workers"]) == 24


def test_missing_duplicate_and_mutated_receipt_fail_before_group_rewrite(synthetic_comparison, monkeypatch):
    flow = synthetic_comparison
    monkeypatch.setattr(parallel, "get_runner", lambda *args: original)
    with pytest.raises(ValueError, match="Every original"):
        parallel.merge_groups(flow.parallel, "validation", flow.results[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        parallel.merge_groups(flow.parallel, "validation", [flow.results[0]] * 24)
    changed = copy.deepcopy(flow.results)
    changed[0]["proof"]["worker_pid"] = -1
    with pytest.raises(ValueError, match="receipt|proof"):
        parallel.merge_groups(flow.parallel, "validation", changed)


def test_exact_task_plan_and_default_dry_plan_need_no_models(monkeypatch, capsys):
    assert len(parallel.plan_cases()) == 24
    assert len(parallel.plan_cases("test", "cap075")) == 12
    assert {p["family"] for p in parallel.plan_cases("test", "cap075")} == {"cap075"}
    for bad in (("train", None), ("validation", "raw"), ("test", None), ("test", "other")):
        with pytest.raises(ValueError):
            parallel.plan_cases(*bad)
    monkeypatch.setattr(sys := parallel.sys, "argv", ["parallel-eval", "--split", "validation"])
    monkeypatch.setattr(parallel, "get_runner", lambda *args: pytest.fail("Planning imported the financial/model runner"))
    parallel.main()
    value = json.loads(capsys.readouterr().out)
    assert value["planning_only"] and value["no_data_or_model_loaded"] and value["case_count"] == 24


@pytest.mark.parametrize("workers", [0, -1, 13, 1.5, True])
def test_invalid_parallelism_does_not_open_data(workers):
    with pytest.raises(ValueError, match="workers"):
        parallel.run_parallel("unread", "unread", "unwritten", workers_per_gpu=workers)


def test_execution_without_new_authorization_does_not_open_data(monkeypatch):
    monkeypatch.setattr(parallel, "get_runner", lambda *args: pytest.fail("Unauthorized execution loaded a runner"))
    with pytest.raises(ValueError, match="Planning does not authorize"):
        parallel.run_parallel("unread", "unread", "unwritten")


def test_existing_case_ownership_refuses_shared_writer_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "synthetic_existing"; output.mkdir()
    original.write_new(output / "protocol.json", {"synthetic_only": True})
    model = ConstantSyntheticPolicy(0, (-1., -1.))
    original.write_new(output / "training" / original.job_id(0) / "record.json",
        dict(seed=0, parameter_sha256=original.parameter_digest(model)))
    case = output / "validation/raw/balanced/base/0"
    case.mkdir(parents=True)
    (case / "sentinel").write_bytes(b"Existing partial account must remain intact")
    monkeypatch.setattr(parallel, "get_runner", lambda *args: original)
    monkeypatch.setattr(original, "protocol", lambda *args: {"family_gpu_assignments": {"raw": "MOCK"}})
    monkeypatch.setattr(original.shared, "torch_setup", lambda *args: {"visible_gpu_uuid": "MOCK", "capability": [7, 0]})
    monkeypatch.setattr(original.shared, "load_dataset", lambda *args: {})
    monkeypatch.setattr(original, "load_policy", lambda *args, **kwargs: model)
    monkeypatch.setattr(original, "evaluate", lambda *args: pytest.fail("Duplicate account reached evaluation"))
    with pytest.raises(FileExistsError):
        parallel.case_worker(dict(parallel.plan_cases()[0], main=str(tmp_path), dataset=str(tmp_path / "unread"),
            output=str(output), protocol_sha256=original.sha(output / "protocol.json"),
            source_sha256=parallel.sha(parallel.__file__)))
    assert list(case.iterdir()) == [case / "sentinel"]
    assert (case / "sentinel").read_bytes() == b"Existing partial account must remain intact"


def test_source_bound_authorization_checks_prerequisite_hash_before_training(tmp_path, monkeypatch):
    output = tmp_path / "synthetic_authorization"; output.mkdir()
    original.write_new(output / "protocol.json", {"synthetic_only": True})
    (output / "prerequisite").write_bytes(b"Synthetic prior scheduler proof")
    authorization = output / "authorization.json"
    value = dict(parallel_case_evaluation_authorized=True, no_live_trading=True,
        financial_parameters_fixed=True, split="validation", family=None,
        source_sha256=parallel.sha(parallel.__file__), original_runner_sha256=parallel.DRIVER_SHA,
        protocol_sha256=original.sha(output / "protocol.json"),
        prerequisite_evidence_sha256={"prerequisite": original.sha(output / "prerequisite")})
    original.write_new(authorization, value)
    checks = []
    monkeypatch.setattr(parallel, "get_runner", lambda *args: original)
    monkeypatch.setattr(original, "protocol", lambda *args: {"synthetic": True})
    monkeypatch.setattr(original, "training_files", lambda *args: checks.append("all5checked"))
    _, actual = parallel.authorize(tmp_path, tmp_path / "unread", output, "validation", None, authorization)
    assert actual == value and checks == ["all5checked"]
    checks.clear()
    (output / "prerequisite").write_bytes(b"Changed prior proof")
    with pytest.raises(ValueError, match="changed"):
        parallel.authorize(tmp_path, tmp_path / "unread", output, "validation", None, authorization)
    assert checks == []


@pytest.mark.parametrize("fail", [False, True])
def test_spawn_plan_uses_12_workers_per_original_family_and_preserves_failure(tmp_path, monkeypatch, fail):
    output = tmp_path / "synthetic_dispatch"; output.mkdir()
    original.write_new(output / "protocol.json", {"synthetic_only": True})
    original.write_new(output / "authorization.json", {"mocked_only": True})
    pools = []
    spec = dict(family_gpu_assignments={"raw": "GPU-0-MOCK", "cap075": "GPU-1-MOCK"})

    class Pool:
        def __init__(self, **kwargs):
            self.kwargs, self.tasks, self.closed = kwargs, [], False
            pools.append(self)

        def submit(self, fn, task):
            assert fn is parallel.case_worker
            self.tasks.append(task)
            future = Future()
            if fail and task["family"] == "raw" and task["key"] == "0":
                future.set_exception(RuntimeError("Explicit synthetic worker failure"))
            else:
                future.set_result({"identity": {k: task[k] for k in ("split", "family", "scenario", "key")}})
            return future

        def shutdown(self, **kwargs):
            assert kwargs == dict(wait=True, cancel_futures=True)
            self.closed = True

    def merge(path, split, results, **kwargs):
        assert len(results) == 24
        value = dict(case_count=24, repeat_count=24, execution_mode="parallel_cases")
        original.write_new(Path(path) / "validation_completion.json", value)
        return value

    monkeypatch.setattr(parallel, "get_runner", lambda *args: original)
    monkeypatch.setattr(parallel, "authorize", lambda *args: (spec, {}))
    monkeypatch.setattr(original, "training_files", lambda *args: None)
    monkeypatch.setattr(parallel, "merge_groups", merge)
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", Pool)
    if fail:
        with pytest.raises(RuntimeError, match="synthetic worker failure"):
            parallel.run_parallel(tmp_path, tmp_path / "unread", output, authorization=output / "authorization.json")
    else:
        parallel.run_parallel(tmp_path, tmp_path / "unread", output, authorization=output / "authorization.json")
    assert len(pools) == 2 and all(pool.closed for pool in pools)
    assert [pool.kwargs["max_workers"] for pool in pools] == [12, 12]
    assert [pool.kwargs["initargs"][1] for pool in pools] == ["GPU-0-MOCK", "GPU-1-MOCK"]
    assert all(len(pool.tasks) == 12 for pool in pools)
    assert all(pool.kwargs["mp_context"].get_start_method() == "spawn" for pool in pools)
    proof = original.read_json(output / "parallel_validation_process.json")
    assert proof["status"] == ("failed" if fail else "completed")
    assert (output / "validation/raw/balanced").is_dir()
    assert (output / "validation/cap075/balanced").is_dir()
    assert (output / "validation_completion.json").exists() is not fail
