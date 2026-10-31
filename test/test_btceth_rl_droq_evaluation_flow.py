"""Synthetic295-day saved-group interoperability, never model/GPU evidence.

Only pytest temporary files are written. The policies are constant CPU tensor
containers, not neural networks or trained checkpoints. GPU setup and policy/
dataset loading are explicitly mocked. f146 evaluation, accounting, serialization,
exact repetition, and bf33 independent financial checks execute unchanged.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import btceth_rl_droq as runner
from scripts import audit_btceth_rl_droq as audit

# The full24-account295-day independent inventory reconstruction is deliberately
# larger than this repository's generic per-test timeout. This changes only the
# synthetic fixture's wall-clock limit, never a financial/audit rule or budget.
pytestmark = pytest.mark.timeout(1200)


class ConstantSyntheticPolicy:
    """No nn.Module construction, sampling, optimizer or device transfer."""
    def __init__(self, seed, action):
        self.seed, self.action, self.calls = seed, np.asarray(action, dtype=np.float32), 0
        self.policy = self
        self.tensor = torch.tensor([float(seed)], dtype=torch.float32, device="cpu")

    def state_dict(self):
        return {"explicitly_synthetic_constant": self.tensor}

    def predict(self, observation, deterministic=True):
        assert deterministic is True and np.asarray(observation).shape == (27,)
        self.calls += 1
        return self.action.copy(), None


def flat_validation_data():
    start, end = map(pd.Timestamp, runner.VALIDATION)
    hours = pd.date_range(start-pd.Timedelta(days=210), end, freq="h", inclusive="left")
    prices = pd.DataFrame({"BTCUSDT": 100., "ETHUSDT": 200.}, index=hours)
    result = {"hourly_"+name: prices.copy() for name in ("open", "high", "low", "close")}
    result["hourly_volume"] = prices*0+100.
    return result


@pytest.fixture(scope="module")
def saved_flow(tmp_path_factory):
    assert not torch.cuda.is_initialized()
    root = tmp_path_factory.mktemp("droq_synthetic_saved_flow")
    output, main = root/"synthetic_output", root/"synthetic_reference"
    output.mkdir(); main.mkdir()
    dataset_path = root/"intentionally_not_a_pickle"; dataset_path.write_bytes(b"No market/checkpoint payload may be read")
    runner.write_new(output/"protocol.json", dict(synthetic_only=True, not_trained_models=True, not_gpu_evidence=True,
        validation=list(runner.VALIDATION), note="Only save/group/accounting interoperability is under test"))
    data = flat_validation_data()
    gpus = [dict(uuid=f"SYNTHETIC-GPU-{i}", name="V100 mocked identity, no CUDA execution") for i in range(2)]
    spec = dict(physical_gpus=gpus, family_gpu_assignments={family:gpus[i]["uuid"] for i,family in enumerate(runner.BUDGETS)})
    actions = [(-1., -1.), (1., -1.), (1., 1.), (.5, 0.), (0., .5)]
    records, models = {}, {}
    for seed, action in enumerate(actions):
        model = ConstantSyntheticPolicy(seed, action); models[seed] = model
        record = dict(algorithm="droq", profile="balanced", seed=seed, explicitly_synthetic_not_trained=True,
            parameter_sha256=runner.parameter_digest(model))
        records[("balanced",seed)] = record
        runner.write_new(output/"training"/runner.job_id(seed)/"record.json", record)
    active = {}; loaded = []; data_loads = []
    def setup(seed):
        return dict(visible_gpu_uuid=spec["family_gpu_assignments"][active["family"]], device="cuda:0",
            name="V100 synthetic setup only; real tensors remain CPU", capability=[7,0], explicitly_mocked=True)
    def load_policy(directory, *, upstream, device):
        assert device == "cuda:0" and upstream == output/"upstream"
        seed = int(Path(directory).name.rsplit("seed",1)[1]); loaded.append(seed)
        return models[seed]
    def load_dataset(path):
        assert path == dataset_path; data_loads.append(path)
        return data
    workers, groups, checks = [], [], audit.Checks()
    with pytest.MonkeyPatch.context() as patch:
        # These are the only mocked execution boundaries. No environment,
        # evaluate/save/metrics, RNG or independent audit function is replaced.
        patch.setattr(runner, "protocol", lambda *args: spec)
        patch.setattr(runner.shared, "torch_setup", setup)
        patch.setattr(runner.shared, "load_dataset", load_dataset)
        patch.setattr(runner, "load_policy", load_policy)
        before = runner.rng_digest()
        for family in runner.BUDGETS:
            active["family"] = family
            workers.append(runner.evaluate_worker(dict(main=str(main), dataset=str(dataset_path), output=str(output), family=family, split="validation")))
        assert before == runner.rng_digest()
        assert loaded == list(range(5))*2 and len(data_loads) == 2
    assert not torch.cuda.is_initialized()
    features = audit.prior.independent_market_features(data, cutoff=runner.VALIDATION[1])
    inputs = audit.Inputs(output, main, root/"unread_source_placeholder", checks)
    for family in runner.BUDGETS:
        groups.append(audit.audit_group(inputs, data, records, spec, split="validation", family=family, features=features))
    assert checks.failed == 0, checks.failures[:15]
    yield SimpleNamespace(output=output, data=data, spec=spec, records=records, models=models, workers=workers, groups=groups, checks=checks)
    assert not torch.cuda.is_initialized()


def test_actual_worker_saved_groups_pass_frozen_bf33_without_nn_or_cuda(saved_flow):
    flow = saved_flow
    assert flow.checks.failed == 0 and flow.checks.count > 100000
    for family, group in zip(runner.BUDGETS, flow.groups):
        summary, daily, monthly, cases = group
        assert len(summary) == len(cases) == 12 and len(daily) == 12*296 and len(monthly) == 12*10
        assert set(summary.scenario) == {"base","stress"} and set(summary.loc[~summary.primary_policy,"seed"]) == set(range(5))
        assert summary.net_return.le(1e-12).all()  # flat invented prices, costs only; no market-performance claim
        cash = summary.loc[summary.seed.eq(0)]
        assert cash.net_return.eq(0).all() and cash.fees.eq(0).all()
        directory = flow.output/"validation"/family/"balanced"
        for case in directory.glob("*/*"):
            if case.is_dir():
                assert set(path.name for path in case.iterdir()) == set(runner.CASE_FILES)
        complete = runner.read_json(directory/"completion.json")
        assert len(complete["repeats"]) == complete["case_count"] == complete["repeat_count"] == 12
        for proof in complete["repeats"]:
            assert proof["rng_sha256_before"] == proof["rng_sha256_after_first"] == proof["rng_sha256_after_repeat"]
            assert proof["parameter_sha256_before"] == proof["parameter_sha256_after"]
            assert all(r["explicitly_synthetic_not_trained"] is True for r in proof["model_records"])
        ensemble = pd.read_csv(directory/"base/ensemble/member_targets.csv", float_precision="round_trip")
        gross = .475*runner.BUDGETS[family]
        np.testing.assert_allclose(ensemble.target_btc+ensemble.target_eth, gross, atol=1e-12, rtol=0)
        assert ensemble.groupby("day").observation_sha256.nunique().eq(1).all()
        assert all(case[0]["hours"] == 7080 and case[0]["reconstructed"] for case in cases.values())
    assert all(model.calls == 4720 for model in flow.models.values())  # 2 families ×2 costs ×2 repeat ×295days ×(single+ensemble)


def test_saved_group_csvs_and_parent_tables_reconcile_to_independent_reconstruction(saved_flow):
    flow = saved_flow
    # No pool/dispatch occurs. These are the same3 parent exports consumed by
    # bf33, assembled solely from the actual frozen worker's saved group CSVs.
    actual, expected = [], []
    for index, name in enumerate(runner.GROUP_FILES[:3]):
        frames = [pd.read_csv(flow.output/"validation"/family/"balanced"/name, float_precision="round_trip") for family in runner.BUDGETS]
        parent = flow.output/"validation"/name; parent.write_bytes(runner.canonical_csv(pd.concat(frames,ignore_index=True)))
        actual.append(pd.read_csv(parent,float_precision="round_trip"))
        expected.append(pd.concat([group[index] for group in flow.groups],ignore_index=True))
    checks = audit.Checks(); inputs = SimpleNamespace(checks=checks)
    audit.compare_tables(inputs,actual,expected,"Synthetic24-account parent table interoperability")
    rebuilt = {run_id:value for group in flow.groups for run_id,value in group[3].items()}
    audit.compare_group_tables(*actual,rebuilt,checks,"Synthetic24-account independent parent accounting")
    assert checks.failed == 0, checks.failures
    assert [len(frame) for frame in actual] == [24,24*296,24*10]
    assert set(actual[0].family) == set(runner.BUDGETS)
