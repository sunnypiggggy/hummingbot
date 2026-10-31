"""CPU synthetic source parity and accounting; no real market or GPU work."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pytest
import torch

from scripts import btceth_rl_droq as d
from scripts import backtest_btceth_rl as old
from scripts.crypto_rl_reporting import result_fingerprints
from test.test_btceth_rl_env import START, dataset


def rng():
    return deepcopy(np.random.get_state()), torch.get_rng_state().clone()


def restore(value):
    np.random.set_state(value[0]); torch.set_rng_state(value[1])


def same_rng(first, second):
    assert first[0][0] == second[0][0] and np.array_equal(first[0][1], second[0][1]) and first[0][2:] == second[0][2:]
    assert torch.equal(first[1], second[1])


def tiny(overrides=None, *, observed=True):
    agent = d.make_agent(device="cpu", overrides={**dict(hidden_sizes=[8, 8], replay_size=64, batch_size=4, start_steps=2), **(overrides or {})})
    return d.observe_agent(agent) if observed else agent


def populate(agent, count=3):
    for step in range(count):
        obs = np.linspace(-.2, .2, 27, dtype=np.float32) + step * .01
        if isinstance(agent.replay_buffer, d.ObservedReplay):
            agent.replay_buffer.pending = dict(terminated=False, truncated=False, date="2023-08-01T00:00Z")
        agent.store_data(obs, np.array([.15, -.2], np.float32), float(step * .01), obs + .02, False)


@pytest.fixture(autouse=True)
def cpu_only():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    yield
    assert not torch.cuda.is_initialized()


def test_declared_single_recipe_and_true_author_boundaries_preserve_old_globals():
    prior = deepcopy(old.PROFILES), deepcopy(old.ALGORITHMS), deepcopy(old.SOURCE_FILES)
    recipe, spec = d.recipe(), d.experiment_spec()
    assert recipe == dict(hidden_sizes=[256, 256], replay_size=1000000, batch_size=256, lr=.0003, gamma=.99, polyak=.995,
        alpha=.2, auto_alpha=True, target_entropy="auto", start_steps=5000, delay_update_steps="auto", utd_ratio=20,
        num_Q=2, num_min=2, q_target_mode="min", policy_update_delay=20, target_drop_rate=.01, layer_norm=True)
    assert spec["initial_effective_alpha"] == .2 and spec["expected_counters"]["random_exploration_actions"] == 5001
    assert spec["expected_counters"]["q_inner_rounds"] == 555360 and spec["timeout_boundaries"] == 1024
    assert spec["model_count"] == 5 and spec["candidate_count"] == 2 and spec["validation_cases"] == spec["validation_repeats"] == 24
    assert all(spec[k] is True for k in d.FLAGS) and spec["official_uniform_crypto_defaults"] is False
    assert prior == (old.PROFILES, old.ALGORITHMS, old.SOURCE_FILES)


def test_normal_author_imports_have_exact_bytes_and_no_rng_side_effect():
    before, path, old_bytecode = rng(), list(sys.path), sys.dont_write_bytecode
    old_cache = sorted(p.as_posix() for p in d.DEFAULT_UPSTREAM.rglob("*.pyc"))
    cls, core = d.load_upstream()
    same_rng(before, rng())
    assert list(sys.path) == path and cls.__module__ == "redq.algos.redq_sac"
    assert sys.dont_write_bytecode is old_bytecode
    assert sorted(p.as_posix() for p in d.DEFAULT_UPSTREAM.rglob("*.pyc")) == old_cache
    assert sys.modules[cls.__module__].ReplayBuffer is core.ReplayBuffer
    assert d.loaded_sources()["OriginalREDQCodebase/redq/algos/core.py"]["sha256"]


def test_initial_alpha_overwrite_target_training_mode_and_actual_regularizers():
    d.seed_cpu(7); agent = tiny()
    assert agent.alpha == .2 and agent.log_alpha.item() == 0 and agent.target_entropy == -2
    params = d.effective_parameters(agent)
    assert params["dropout_per_q"] == [[.01, .01], [.01, .01]]
    assert params["layer_norm_per_q"] == [[[8], [8]], [[8], [8]]]
    assert params["target_training"] == [True, True]
    assert all(p.requires_grad for q in agent.q_target_net_list for p in q.raw.parameters())


def test_first_actual_update_matches_unchanged_author_source_and_same_rng():
    d.seed_cpu(11); reference = tiny(observed=False)
    d.seed_cpu(11); observed = tiny()
    assert d.parameter_digest(reference) == d.parameter_digest(observed)
    populate(reference); populate(observed)
    before = rng(); reference.train(d.MetricSink()); after_reference = rng()
    restore(before)
    alpha_at_targets = []
    original_target = observed.get_redq_q_target_no_grad
    def traced(*args, **kwargs):
        alpha_at_targets.append(observed.alpha)
        return original_target(*args, **kwargs)
    # An instance-only, read-only fixture observation; no upstream globals or
    # mathematics are changed, and RNG equality below catches extra sampling.
    observed.get_redq_q_target_no_grad = traced
    d.original_train(observed, d.MetricSink())
    same_rng(after_reference, rng())
    assert alpha_at_targets == [.2] * 20
    assert observed.alpha == reference.alpha == pytest.approx(float(observed.log_alpha.exp().item()))
    assert observed.alpha > .99 and d.parameter_digest(reference) == d.parameter_digest(observed)
    for left, right in zip(reference.q_optimizer_list, observed.q_optimizer_list):
        assert d.optimizer_steps(left) == d.optimizer_steps(right) == 20
        for lstate, rstate in zip(left.state.values(), right.state.values()):
            for name in lstate:
                assert torch.equal(lstate[name], rstate[name])
    counts = d.actual_counters(observed, environment_steps=3, random_actions=3)
    assert counts == dict(q_inner_rounds=20, q_optimizer_steps=[20, 20], total_q_optimizer_steps=40,
        policy_optimizer_steps=1, alpha_optimizer_steps=1, target_updates_per_network=[20, 20], learning_train_calls=1,
        environment_steps=3, replay_adds=3, random_exploration_actions=3)


def test_replay_observer_reads_real_arrays_and_does_not_change_sampling_or_rng():
    _, core = d.load_upstream()
    plain, wrapped = core.ReplayBuffer(27, 2, 5), d.ObservedReplay(core.ReplayBuffer(27, 2, 5))
    obs, terminal = np.zeros(27, np.float32), np.full(27, .25, np.float32)
    before = rng()
    plain.store(obs, [0, 0], .1, terminal, False)
    wrapped.pending = dict(terminated=False, truncated=True, date="2023-08-02T00:00Z")
    wrapped.store(obs, [0, 0], .1, terminal, False)
    same_rng(before, rng())
    for name in ("obs1_buf", "obs2_buf", "acts_buf", "rews_buf", "done_buf"):
        np.testing.assert_array_equal(getattr(plain, name), getattr(wrapped.raw, name))
    row = wrapped.boundaries[0]
    assert row["done"] == row["timeout"] == 1 and row["stored_done"] == row["effective_done"] == 0
    assert row["time_limit_truncated"] is True and row["terminal_observation_sha256"] == row["stored_next_observation_sha256"]
    before = rng(); expected = plain.sample_batch(4); after = rng(); restore(before)
    actual = wrapped.sample_batch(4); same_rng(after, rng())
    for name in expected:
        np.testing.assert_array_equal(expected[name], actual[name])


class SyntheticEnv(gym.Env):
    def __init__(self, length=32, *, economic_done=False):
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-10, 10, shape=(27,), dtype=np.float32)
        self.length, self.economic_done, self.resets = length, economic_done, 0
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.at = 0; self.resets += 1
        return np.full(27, -.5, np.float32), {}
    def step(self, action):
        self.at += 1; end = self.at == self.length
        return np.full(27, self.at / 100, np.float32), .01, end and self.economic_done, end and not self.economic_done, {"date": "2023-08-02T00:00Z"}


def test_bounded_driver_keeps_terminal_not_reset_and_has_no_final_unused_reset():
    d.seed_cpu(3); agent = tiny(dict(start_steps=100))
    env = SyntheticEnv(); result = d.drive_steps(agent, env, 64, seed=3)
    assert result["counters"]["environment_steps"] == result["counters"]["replay_adds"] == 64
    assert result["counters"]["random_exploration_actions"] == 64 and result["counters"]["q_inner_rounds"] == 0
    assert env.resets == 2 and len(agent.replay_buffer.boundaries) == 2
    independent = spaces.Box(-1, 1, shape=(2,), dtype=np.float32); independent.seed(3)
    assert result["action_space_rng_before"] == independent.np_random.bit_generator.state
    for _ in range(64):
        independent.sample()
    assert result["action_space_rng_after"] == independent.np_random.bit_generator.state
    for row in agent.replay_buffer.boundaries:
        assert row["terminal_observation_values"] == row["stored_next_observation_values"]
        assert row["terminal_observation_values"][0] == pytest.approx(.32)
        assert row["stored_done"] == 0


def test_economic_termination_is_not_rewritten_as_timeout():
    d.seed_cpu(3); agent = tiny(dict(start_steps=100)); env = SyntheticEnv(length=2, economic_done=True)
    d.drive_steps(agent, env, 2, seed=3)
    row = agent.replay_buffer.boundaries[0]
    assert row["done"] == row["stored_done"] == row["effective_done"] == 1 and row["timeout"] == 0


def test_strict_after_store_boundary_matches_original_no_learning_at_two_then_three():
    d.seed_cpu(2); agent = tiny(); result = d.drive_steps(agent, SyntheticEnv(length=32), 3, seed=2)
    assert result["counters"]["random_exploration_actions"] == 3
    assert result["counters"]["learning_train_calls"] == 1 and result["counters"]["q_inner_rounds"] == 20
    assert result["counters"]["policy_optimizer_steps"] == result["counters"]["alpha_optimizer_steps"] == 1


@pytest.fixture
def checkpoint(tmp_path):
    d.seed_cpu(5)
    agent = d.observe_agent(d.make_agent(device="cpu", overrides=dict(replay_size=64, batch_size=4, start_steps=2)))
    populate(agent); d.original_train(agent, d.MetricSink())
    counters = d.actual_counters(agent, environment_steps=3, random_actions=3)
    before = rng(); manifest = d.save_checkpoint(tmp_path / "model", agent, counters=counters); same_rng(before, rng())
    return tmp_path / "model", agent, manifest


def test_checkpoint_is_tensor_json_only_and_optimizer_steps_cover_every_parameter(checkpoint):
    folder, agent, manifest = checkpoint
    before = rng(); loaded, tensors = d.read_checkpoint(folder); same_rng(before, rng())
    assert len(loaded["tensor_groups"]) == 13
    assert loaded["parameter_sha256"] == d.parameter_digest(agent)
    for group, count, step in (("optimizer_policy", 8, 1), ("optimizer_q0", 10, 20), ("optimizer_q1", 10, 20), ("optimizer_alpha", 1, 1)):
        proof = loaded["tensor_groups"][group]
        assert len(proof["step_tensor_keys"]) == count and proof["step_values"] == [float(step)] * count
    assert set(tensors) == set(loaded["tensors"])
    assert all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in tensors.items())


def test_loaded_policy_matches_author_mean_action_without_consuming_rng(checkpoint):
    folder, agent, manifest = checkpoint
    before = rng(); policy = d.load_policy(folder, device="cpu"); same_rng(before, rng())
    values = np.linspace(-.1, .1, 27, dtype=np.float32)
    expected = agent.get_test_action(values); action, state = policy.predict(values, deterministic=True)
    same_rng(before, rng())
    np.testing.assert_array_equal(action, expected)
    assert state is None and d.parameter_digest(policy) == manifest["parameter_sha256"]
    with pytest.raises(ValueError, match="deterministic"):
        policy.predict(values, deterministic=False)


def test_independent_auditor_can_read_actual_tensor_groups_without_agent_or_rng(checkpoint):
    from scripts import audit_btceth_rl_droq as audit
    folder, agent, manifest = checkpoint
    expected = audit.update_counts(3, start_steps=2, delay_update_steps=2)
    before = rng()
    proof = audit.read_tensor_checkpoint(folder / "model_tensors.pt", folder / "model.json", expected_counters=expected)
    same_rng(before, rng())
    assert proof["parameter_sha256"] == manifest["parameter_sha256"] == d.parameter_digest(agent)
    assert proof["actual_counters"] == expected and proof["cuda_initialized_changed"] is False


def test_checkpoint_tampering_is_rejected(checkpoint):
    folder = checkpoint[0]; path = folder / "model.json"; value = json.loads(path.read_text())
    value["counters"]["policy_optimizer_steps"] += 1; path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="counter"):
        d.read_checkpoint(folder)


class State:
    def state_dict(self):
        return {"synthetic": torch.tensor([1.])}


class Fixed:
    def __init__(self, action):
        self.action, self.policy, self.observed = action, State(), []
    def predict(self, obs, deterministic=True):
        self.observed.append(obs.copy())
        return np.asarray(self.action(obs) if callable(self.action) else self.action, dtype=np.float32), None


def test_five_target_ensemble_uses_one_actual_cash_ledger_and_common_feedback():
    models = [Fixed([1, -1]) for _ in range(3)] + [Fixed([0, 1]) for _ in range(2)]
    result, trace = d.evaluate(dataset(), models, list(range(5)), "cap075", START, START + pd.Timedelta(days=2), d.COSTS["base"])
    np.testing.assert_allclose(trace.target_btc, .4275, rtol=0, atol=1e-15)
    assert len(result["fills"]) == 1 and result["fills"].asset_key.tolist() == ["BTCUSDT"]
    assert all(np.array_equal(m.observed[1], models[0].observed[1]) for m in models)
    assert models[0].observed[1][-4] == pytest.approx(.4275)
    assert result["metadata"]["parameter_sha256_before"] == result["metadata"]["parameter_sha256_after"]


def test_costs_and_original_sell_then_next_hour_buy_are_not_changed():
    model = Fixed(lambda obs: [1, -1] if obs[-4] == 0 else [1, 1])
    result, _ = d.evaluate(dataset(), [model], [0], "raw", START, START + pd.Timedelta(days=2), d.COSTS["stress"])
    fills = result["fills"]
    assert fills.side.tolist() == ["BUY", "SELL", "BUY"] and fills.timestamp.iloc[2] - fills.timestamp.iloc[1] == pd.Timedelta(hours=1)
    assert result["equity"].cash.ge(0).all() and result["equity"].base_value.ge(0).all()


def test_future_hourly_tail_is_removed_before_features_and_history_is_unchanged():
    data = dataset(); cutoff = START + pd.Timedelta(days=2); changed = deepcopy(data)
    for key in ("hourly_open", "hourly_high", "hourly_low", "hourly_close"):
        changed[key].loc[cutoff:] *= 99
    one = d.evaluate(d.cropped_data(data, cutoff), [Fixed([1, -1])], [0], "raw", START, cutoff, d.COSTS["base"])
    two = d.evaluate(d.cropped_data(changed, cutoff), [Fixed([1, -1])], [0], "raw", START, cutoff, d.COSTS["base"])
    assert result_fingerprints(one[0]) == result_fingerprints(two[0])
    pd.testing.assert_frame_equal(one[1], two[1])


@pytest.fixture
def source_stage(tmp_path, monkeypatch):
    """SHA-only fake bytes: never a real market pickle or formal output."""
    source, main, data_dir = (tmp_path / name for name in ("source", "original", "data"))
    main.mkdir(); data_dir.mkdir()
    actual_root = d.ROOT
    for name in d.SOURCE_FILES:
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual_root / name, path)
    monkeypatch.setattr(d, "ROOT", source)
    monkeypatch.setattr(d, "financial_imports", lambda: None)
    data = data_dir / "dataset.pkl.gz"; data.write_bytes(b"synthetic SHA fixture, deliberately not a pickle")
    (data_dir / "manifest.json").write_text("{}")
    gpus = [dict(index=i, uuid=f"GPU-fixture{i}", name="Tesla V100-SXM2-32GB", memory_total_mib=32768,
        memory_used_mib=10, utilization_percent=0) for i in range(2)]
    original = dict(timesteps_per_model=d.STEPS, train=list(d.TRAIN), validation=list(d.VALIDATION), test=list(d.TEST),
        research_targets=d.TARGETS, runtime_packages=d.shared.RUNTIME_PACKAGES,
        dataset_sha256=d.sha(data), dataset_manifest_sha256=d.sha(data_dir / "manifest.json"), code_sha256=d.pins(source, d.shared.SOURCE_FILES))
    d.write_new(main / "frozen_spec.json", original); d.write_new(main / "runtime.json", dict(physical_gpus=gpus))
    for name in d.baseline_files():
        path = main / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"synthetic original reference")
    old_inputs = d.pins(main, d.baseline_files() | {"frozen_spec.json", "runtime.json"})
    d.write_new(main / "independent_audit.json", dict(passed=True, status="passed", test_complete=True, run_count=172, failures=[], input_sha256=old_inputs))
    auth = tmp_path / "approval.json"
    d.write_new(auth, dict(authorized_at="2026-10-01T00:00:00Z", fit_validation_authorized=True, no_live_trading=True,
        reviewed_source_sha256=d.pins(source, d.SOURCE_FILES), hyperparameters=d.recipe(), seeds=list(d.SEEDS),
        timesteps_per_model=d.STEPS, budgets=d.BUDGETS, **{key: True for key in d.FLAGS}))
    return dict(main=main, dataset=data, output=tmp_path / "new_study", authorization=auth,
        upstream_root=d.DEFAULT_UPSTREAM, config_contract=d.DEFAULT_UPSTREAM.parents[1] / "effective_config_contract.json")


def prepare_stage(stage):
    return d.prepare(stage["main"], stage["dataset"], stage["output"], **{k: stage[k] for k in ("authorization", "upstream_root", "config_contract")})


def test_prepare_protocol_and_source_binding_are_sha_only_and_fail_closed(source_stage, monkeypatch):
    monkeypatch.setattr(d.shared, "load_dataset", lambda *_: pytest.fail("SHA-only preparation cannot read market payloads"))
    value = prepare_stage(source_stage)
    assert d.protocol(source_stage["main"], source_stage["dataset"], source_stage["output"]) == value
    assert value["seed_gpu_assignments"] == {str(s): f"GPU-fixture{s % 2}" for s in d.SEEDS}
    assert value["family_gpu_assignments"] == dict(raw="GPU-fixture0", cap075="GPU-fixture1")
    assert value["environment_step_budget_multiple"] == 1 and value["compute_budget_comparable"] is False
    assert "actor-only" in value["effective_contract_refinement"]
    assert not (source_stage["output"] / "training").exists()
    with pytest.raises(FileExistsError):
        prepare_stage(source_stage)
    file = source_stage["output"] / "sources/scripts/btceth_rl_droq.py"
    file.write_bytes(file.read_bytes() + b"\n# tampered")
    with pytest.raises(ValueError, match="changed"):
        d.protocol(source_stage["main"], source_stage["dataset"], source_stage["output"])


def test_prepare_rejects_unapproved_recipe_before_creating_output(source_stage):
    value = d.read_json(source_stage["authorization"]); value["hyperparameters"]["target_drop_rate"] = .05
    source_stage["authorization"].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="authorization"):
        prepare_stage(source_stage)
    assert not source_stage["output"].exists()


def test_train_one_consumes_driver_counters_and_preserves_producer_schema(source_stage, monkeypatch):
    """Orchestration only; real mathematical training parity is tested above."""
    from scripts import btceth_rl_env
    value = prepare_stage(source_stage); output = source_stage["output"]
    device = dict(visible_gpu_uuid="GPU-fixture0", device="cuda:0", name="Tesla V100", capability=[7, 0])
    monkeypatch.setattr(d.shared, "torch_setup", lambda seed: device)
    synthetic = {"hourly_close": pd.DataFrame({"BTCUSDT": [1., 2.], "ETHUSDT": [1., 2.]},
        index=pd.DatetimeIndex([pd.Timestamp(d.TRAIN[1]) - pd.Timedelta(hours=1), pd.Timestamp(d.TRAIN[1])]))}
    monkeypatch.setattr(d.shared, "load_dataset", lambda path: synthetic)
    class Env:
        FEATURE_NAMES = tuple(f"fixture{i}" for i in range(27))
        training_episode_history = [dict(synthetic_only=True)]
        training_window_metadata = dict(synthetic_only=True)
        def __init__(self, data, *dates, **kw):
            assert dates == d.TRAIN and kw["episode_days"] == 32 and kw["training"] is True
            assert data["hourly_close"].index.max() < pd.Timestamp(d.TRAIN[1])
        def close(self):
            self.closed = True
    monkeypatch.setattr(btceth_rl_env, "BTCETHPortfolioEnv", Env)
    FakeAgent = type("REDQSACAgent", (), {"__module__": "redq.algos.redq_sac"})
    agent = FakeAgent(); agent.alpha, agent.log_alpha = .2, torch.tensor([0.])
    monkeypatch.setattr(d, "make_agent", lambda *args, **kw: agent)
    def observed(obj, trace):
        trace.write_text("{}\n" * 1024); return obj
    monkeypatch.setattr(d, "observe_agent", observed)
    expected = d.experiment_spec()["expected_counters"]
    def drive(obj, env, steps, **kw):
        assert steps == 32768 and kw["seed"] == 0
        return dict(counters=expected, transition_sha256="fixture_transition", action_space_rng_before={"fixture": 0}, action_space_rng_after={"fixture": 1})
    monkeypatch.setattr(d, "drive_steps", drive)
    def save(folder, obj, *, counters, upstream):
        assert counters == expected
        (folder / "model_tensors.pt").write_bytes(b"not a real checkpoint: orchestration fixture")
        d.write_new(folder / "model.json", dict(synthetic_only=True))
        return dict(parameter_sha256="fixture_parameter")
    monkeypatch.setattr(d, "save_checkpoint", save)
    monkeypatch.setattr(d, "effective_parameters", lambda _: dict(synthetic_only=True))
    monkeypatch.setattr(d, "loaded_sources", lambda _: dict(synthetic_only=True))
    record = d.train_one(dict(main=str(source_stage["main"]), dataset=str(source_stage["dataset"]), output=str(output), profile="balanced", seed=0))
    assert record["actual_counters"] == expected and record["actual_timesteps"] == 32768
    assert record["action_space_rng_before"] == {"fixture": 0} and record["action_space_rng_after"] == {"fixture": 1}
    assert record["actual_transition_sha256"] == "fixture_transition" and len(record["features"]) == 27
    assert record["model_tensor_file"] == "training/droq__balanced__seed0/model_tensors.pt"
    assert record["protocol_sha256"] == d.sha(output / "protocol.json")


def test_supervisor_true_child_terminal_and_failed_attempt_are_preserved(source_stage, monkeypatch):
    prepare_stage(source_stage); output = source_stage["output"]
    monkeypatch.setattr(d, "gpu_preflight", lambda spec: dict(at=d.now(), physical_gpus=spec["physical_gpus"]))
    commands = []
    class Child:
        pid = 987654
        def __init__(self, command, **kw):
            commands.append(command); kw["stdout"].write("synthetic child only\n")
        def wait(self):
            proof = d.read_json(output / "fit_process.json")
            assert proof["status"] == "running" and proof["child_pid"] == self.pid
            d.write_new(output / "fit_completion.json", dict(at=d.now(), model_count=5, protocol_sha256=d.sha(output / "protocol.json")))
            return 0
    monkeypatch.setattr(d.subprocess, "Popen", Child)
    process = d.supervise(source_stage["main"], source_stage["dataset"], output, "fit")
    assert process["status"] == "completed" and process["exit_code"] == 0 and process["child_pid"] == Child.pid
    assert commands[0][commands[0].index("--phase") + 1] == "_fit" and d.terminal(output, "fit") == process
    with pytest.raises(FileExistsError):
        d.supervise(source_stage["main"], source_stage["dataset"], output, "fit")
    process["exit_code"] = False; (output / "fit_process.json").write_text(json.dumps(process))
    with pytest.raises(ValueError, match="completed0"):
        d.terminal(output, "fit")


def test_failed_child_preserves_original_exit_and_log_without_a_completion(source_stage, monkeypatch):
    prepare_stage(source_stage); output = source_stage["output"]
    monkeypatch.setattr(d, "gpu_preflight", lambda spec: {"at": d.now()})
    class Failed:
        pid = 987655
        def __init__(self, command, **kw):
            kw["stdout"].write("synthetic failed child evidence\n")
        def wait(self):
            return 7
    monkeypatch.setattr(d.subprocess, "Popen", Failed)
    with pytest.raises(RuntimeError, match="Phase failed"):
        d.supervise(source_stage["main"], source_stage["dataset"], output, "fit")
    proof = d.read_json(output / "fit_process.json")
    assert proof["status"] == "failed" and proof["exit_code"] == 7 and proof["child_pid"] == Failed.pid
    assert proof["log_sha256"] == d.sha(output / "fit.log") and not (output / "fit_completion.json").exists()
    with pytest.raises(FileExistsError):
        d.supervise(source_stage["main"], source_stage["dataset"], output, "fit")


def test_private_child_checks_actual_parent_without_launching_any_process(tmp_path, monkeypatch):
    d.write_new(tmp_path / "protocol.json", {"synthetic": True})
    proof = dict(phase="fit", status="running", child_pid=111, supervisor_pid=222,
        source_sha256=d.sha(Path(d.__file__)), protocol_sha256=d.sha(tmp_path / "protocol.json"))
    d.write_new(tmp_path / "fit_process.json", proof)
    monkeypatch.setattr(d.os, "getpid", lambda: 111); monkeypatch.setattr(d.os, "getppid", lambda: 222)
    d.private_gate(tmp_path, "fit")
    monkeypatch.setattr(d.os, "getppid", lambda: 333)
    with pytest.raises(ValueError, match="PPID"):
        d.private_gate(tmp_path, "fit")


def validation_frame(family, gain=.1):
    rows = []
    for cost in d.COSTS:
        for seed in (*d.SEEDS, None):
            key = "ensemble" if seed is None else str(seed)
            rows.append(dict(run_id=f"validation__{cost}__{family}__droq__balanced__{key}", scenario=cost,
                primary_policy=seed is None, seed=seed, net_return=gain, max_drawdown=-.1, final_equity=10000*(1+gain), fees=10.))
    return pd.DataFrame(rows)


def test_two_candidate_selection_seals_all_seeds_and_raw_wins_ties(tmp_path, monkeypatch):
    for family in d.BUDGETS:
        folder = tmp_path / "validation" / family / "balanced"; folder.mkdir(parents=True)
        (folder / "summary.csv").write_bytes(d.canonical_csv(validation_frame(family)))
    d.write_new(tmp_path / "protocol.json", {"synthetic": True})
    d.write_new(tmp_path / "independent_audit.validation.json", {"synthetic": True})
    (tmp_path / "all_model_identity").write_bytes(b"all five models fixture")
    val_files = {f"validation/{family}/balanced/summary.csv" for family in d.BUDGETS}
    monkeypatch.setattr(d, "protocol", lambda *_: {"synthetic": True})
    monkeypatch.setattr(d, "terminal", lambda *_: None)
    monkeypatch.setattr(d, "training_files", lambda root: d.pins(root, {"all_model_identity"}))
    monkeypatch.setattr(d, "validation_audit", lambda *_: None)
    monkeypatch.setattr(d, "validation_files", lambda: val_files)
    choice = d.select(tmp_path / "unused_original", tmp_path / "unused_data", tmp_path)
    assert len(choice["candidates"]) == 2 and choice["candidate"]["family"] == "raw"
    assert choice["seed_selection"] is False and choice["test_results_available_to_selector"] is False
    assert d.frozen_selection(None, None, tmp_path)[1] == choice
    with pytest.raises(FileExistsError):
        d.select(None, None, tmp_path)
    path = tmp_path / "validation/cap075/balanced/summary.csv"; path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        d.frozen_selection(None, None, tmp_path)


def test_annual_gate_requires_separate_later_exact_selection_authorization(tmp_path, monkeypatch):
    d.write_new(tmp_path / "protocol.json", {"synthetic": True}); d.write_new(tmp_path / "selection.json", {"synthetic": True})
    seal = dict(sealed_at="2026-10-01T01:00:00Z")
    monkeypatch.setattr(d, "frozen_selection", lambda *_: ({}, {"candidate": {"family": "raw"}}, seal))
    monkeypatch.setattr(d, "terminal", lambda *_: None)
    with pytest.raises(ValueError, match="Separate explicit annual"):
        d.annual_gate(None, None, tmp_path, None, require_new=True)
    auth = tmp_path / "separate_authorization.json"
    value = dict(authorized_at="2026-10-01T02:00:00Z", annual_evaluation_authorized=True, no_live_trading=True,
        financial_parameters_fixed=True, selection_sha256=d.sha(tmp_path / "selection.json"), **{key: True for key in d.FLAGS})
    d.write_new(auth, value)
    assert d.annual_gate(None, None, tmp_path, auth, require_new=True)[1]["candidate"]["family"] == "raw"
    value["authorized_at"] = "2026-10-01T00:00:00Z"; auth.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Annual authorization"):
        d.annual_gate(None, None, tmp_path, auth, require_new=True)
    (tmp_path / "test").mkdir()
    with pytest.raises(ValueError, match="annual attempt"):
        d.no_test(tmp_path)


def test_gpu_identity_uses_static_identity_and_never_filters_other_processes(monkeypatch):
    gpus = [dict(index=i, uuid=f"GPU-{i}", name="Tesla V100", memory_total_mib=32768, memory_used_mib=10, utilization_percent=0) for i in range(2)]
    spec = dict(physical_gpus=gpus, runtime_packages={"fixture": "1"}, allowed_compute_instance=None, allowed_gpu_rows=[])
    current = deepcopy(gpus); current[0].update(memory_used_mib=99, utilization_percent=1)
    monkeypatch.setattr(d.shared, "gpu_inventory", lambda: current)
    monkeypatch.setattr(d.shared, "runtime_packages", lambda: spec["runtime_packages"])
    monkeypatch.setattr(d.subprocess, "check_output", lambda *args, **kw: "")
    assert d.gpu_preflight(spec)["physical_gpus"] == current
    monkeypatch.setattr(d.subprocess, "check_output", lambda *args, **kw: "GPU-0, 12345\n")
    with pytest.raises(RuntimeError, match="Unknown existing GPU process"):
        d.gpu_preflight(spec)
    instance = dict(pid=12345, uid=1000, exe="/usr/bin/ptyxis", starttime_ticks="1", boot_id="a")
    spec.update(allowed_compute_instance=instance, allowed_gpu_rows=[dict(gpu_uuid="GPU-0", pid=12345)])
    d.verify_resource_rows(spec["allowed_gpu_rows"], spec, inspect_process=lambda _: instance)
    with pytest.raises(RuntimeError):
        d.verify_resource_rows(spec["allowed_gpu_rows"], spec, inspect_process=lambda _: {**instance, "starttime_ticks": "2"})


def test_checkpoint_cannot_overwrite_existing_evidence(checkpoint):
    directory, agent, _ = checkpoint
    before = d.sha(directory / "model_tensors.pt"), d.sha(directory / "model.json")
    with pytest.raises(FileExistsError):
        d.save_checkpoint(directory, agent, counters=d.actual_counters(agent))
    assert before == (d.sha(directory / "model_tensors.pt"), d.sha(directory / "model.json"))
