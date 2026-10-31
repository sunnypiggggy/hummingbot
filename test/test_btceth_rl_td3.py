"""CPU synthetic lifecycle, timeout, accounting and causal TD3 fixtures.

No training, learned-policy inference, official preparation or annual prices.
"""
from copy import deepcopy
import json
import importlib.metadata
import os

import numpy as np
import pandas as pd
import pytest

from scripts import backtest_btceth_rl as original
from scripts import btceth_rl_td3 as td3
from scripts.crypto_rl_reporting import result_fingerprints
from test.test_btceth_rl_env import START, dataset


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.value


class FixedPolicy:
    def state_dict(self):
        return {"fixture": Tensor([1., 2.])}


class FixedModel:
    def __init__(self, action):
        self.policy, self.action, self.observations = FixedPolicy(), action, []
    def predict(self, observation, deterministic):
        assert deterministic is True
        self.observations.append(observation.copy())
        action = self.action(observation) if callable(self.action) else self.action
        return np.asarray(action, dtype=np.float32), None


def test_unique_recipe_matches_shared_skeleton_without_foreign_td3_arguments():
    old = deepcopy(original.PROFILES), deepcopy(original.ALGORITHMS), deepcopy(original.SOURCE_FILES)
    recipe = td3.recipe()
    assert recipe["learning_rate"] == .0003 and recipe["gamma"] == .995
    assert (recipe["buffer_size"], recipe["learning_starts"], recipe["batch_size"]) == (100000, 512, 128)
    assert recipe["policy_kwargs"]["net_arch"] == [128, 128]
    assert (recipe["policy_delay"], recipe["target_policy_noise"], recipe["target_noise_clip"]) == (2, .2, .5)
    assert recipe["n_steps"] == 1 and recipe["train_freq"] == [1, "step"]
    assert recipe["replay_buffer_kwargs"] == {"handle_timeout_termination": True}
    assert recipe["action_noise"]["sigma"] == [.1, .1] and recipe["action_noise"]["dtype"] == "float32"
    assert not {"ent_coef", "n_quantiles", "top_quantiles_to_drop_per_net"}.intersection(recipe)
    recipe["policy_kwargs"]["net_arch"][0] = 1
    assert td3.recipe()["policy_kwargs"]["net_arch"] == [128, 128]
    spec = td3.shared_spec()
    assert spec["model_count"] == 10 and spec["validation_cases"] == spec["validation_repeats"] == 48
    assert spec["timesteps_per_model"] == 65536 == spec["old_timesteps_per_model"] * 2
    assert spec["recipe_is_pure_official_default"] is False
    assert all(spec[flag] is True for flag in td3.FLAGS)
    assert (original.PROFILES, original.ALGORITHMS, original.SOURCE_FILES) == old


def test_pinned_official_library_identity_is_read_without_importing_a_model():
    try:
        version = importlib.metadata.version("stable-baselines3")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Pinned SB3 distribution unavailable in this test environment")
    if version != "2.9.0":
        pytest.skip("Formal source identity requires exactly SB3 2.9.0")
    values = td3.library_sources()
    assert len(values) == 5 and all(name.startswith("stable_baselines3/") for name in values)
    assert values["stable_baselines3/td3/td3.py"] == td3.UPSTREAM["td3_sha256"]


@pytest.mark.parametrize("family,gross", [("raw", .95), ("cap075", .7125)])
def test_single_real_account_entry_cost_and_cash_reserve_by_hand(family, gross):
    result, trace = td3.evaluate(dataset(), [FixedModel([1., -1.])], [0], "return", family,
        START, START + pd.Timedelta(days=1), td3.COSTS["base"])
    fee, slip = .001, .0002
    cost = (1 + fee) * (1 + slip) - 1
    nav = 10000 / (1 + gross * cost)
    assert result["metadata"]["final_equity"] == pytest.approx(nav)
    assert result["metadata"]["final_cash"] == pytest.approx((1 - gross) * nav)
    assert result["metadata"]["fees"] == pytest.approx(gross * nav * (1 + slip) * fee)
    assert result["fills"].side.tolist() == ["BUY"]
    assert result["fills"].timestamp.tolist() == [START + pd.Timedelta(hours=1)]
    assert trace.mapped_btc.iloc[0] == pytest.approx(gross)
    assert trace.raw_btc.iloc[0] == .95 and trace.target_cash.iloc[0] == pytest.approx(1 - gross)
    assert result["metadata"]["old_actions_or_equity_scaled"] is False


def test_five_targets_mean_is_not_mean_action_or_mean_of_five_return_curves():
    models = [FixedModel([1., -1.]) for _ in range(3)] + [FixedModel([0., 1.]) for _ in range(2)]
    result, trace = td3.evaluate(dataset(), models, list(td3.SEEDS), "balanced", "cap075",
        START, START + pd.Timedelta(days=2), td3.COSTS["base"])
    assert len(trace) == 10
    np.testing.assert_allclose(trace.target_btc, .4275, atol=1e-15, rtol=0)
    assert trace.target_eth.eq(0).all()
    assert len(result["fills"]) == 1 and result["fills"].asset_key.tolist() == ["BTCUSDT"]
    for index in (0, 1):
        assert all(np.array_equal(model.observations[index], models[0].observations[index]) for model in models)
    assert models[0].observations[0][-4] == 0
    assert models[0].observations[1][-4] == pytest.approx(.4275)
    assert models[0].observations[1][-2] == pytest.approx(.5725)
    assert result["metadata"]["model_count"] == 5
    assert result["metadata"]["parameter_sha256_before"] == result["metadata"]["parameter_sha256_after"]
    wrong = np.mean([[1, -1]] * 3 + [[0, 1]] * 2, axis=0)
    from scripts.btceth_rl_env import BTCETHPortfolioEnv
    env = BTCETHPortfolioEnv(dataset(), START, START + pd.Timedelta(days=1))
    assert not np.allclose(td3.map_target(env.action_to_weights(wrong), "cap075")[:2], [.4275, 0])


def test_actual_ledger_feedback_causes_sell_then_next_hour_buy_without_borrowing():
    model = FixedModel(lambda obs: [1, -1] if obs[-4] == 0 else [1, 1])
    result, trace = td3.evaluate(dataset(), [model], [0], "return", "cap075", START,
        START + pd.Timedelta(days=2), td3.COSTS["stress"])
    fills = result["fills"]
    assert fills.side.tolist() == ["BUY", "SELL", "BUY"]
    assert fills.timestamp.iloc[1] == START + pd.Timedelta(days=1)
    assert fills.timestamp.iloc[2] == START + pd.Timedelta(days=1, hours=1)
    assert fills.asset_key.tolist() == ["BTCUSDT", "BTCUSDT", "ETHUSDT"]
    assert result["equity"].cash.ge(0).all() and result["equity"].base_value.ge(0).all()
    assert trace.target_btc.iloc[0] > 0 and trace.target_eth.iloc[1] > 0


def test_future_tail_is_removed_before_feature_construction_and_original_not_mutated():
    data = dataset(); cutoff = START + pd.Timedelta(days=2)
    before = data["hourly_close"].copy()
    cut = td3.cropped_data(data, cutoff)
    assert cut["hourly_close"].index.max() == cutoff - pd.Timedelta(hours=1)
    cut["hourly_close"].iloc[0, 0] = 999
    pd.testing.assert_frame_equal(data["hourly_close"], before)
    altered = deepcopy(data)
    for field in ("open", "high", "low", "close"):
        altered["hourly_" + field].loc[cutoff:] *= 100
    models = [FixedModel([1., -1.]), FixedModel([1., -1.])]
    first = td3.evaluate(td3.cropped_data(data, cutoff), [models[0]], [0], "return", "raw", START, cutoff, td3.COSTS["base"])
    second = td3.evaluate(td3.cropped_data(altered, cutoff), [models[1]], [0], "return", "raw", START, cutoff, td3.COSTS["base"])
    assert result_fingerprints(first[0]) == result_fingerprints(second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_replay_observer_reads_real_super_add_without_rng_or_array_mutation(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    import torch
    from gymnasium import spaces
    from stable_baselines3.common.buffers import ReplayBuffer
    assert not torch.cuda.is_initialized()
    kwargs = dict(buffer_size=4, observation_space=spaces.Box(-10, 10, shape=(27,), dtype=np.float32),
        action_space=spaces.Box(-1, 1, shape=(2,), dtype=np.float32), device="cpu", n_envs=1,
        optimize_memory_usage=False, handle_timeout_termination=True)
    path = tmp_path / "timeouts.jsonl"
    observed, plain = td3.audited_replay_buffer_class(path)(**kwargs), ReplayBuffer(**kwargs)
    obs, end = np.zeros((1, 27), np.float32), np.full((1, 27), .25, np.float32)
    action, reward, done = np.zeros((1, 2), np.float32), np.array([1.]), np.array([True])
    infos = [{"TimeLimit.truncated": True, "terminal_observation": end[0], "date": "2023-08-02T00:00:00Z"}]
    rng = np.random.get_state()
    observed.add(obs, end, action, reward, done, infos); plain.add(obs, end, action, reward, done, infos)
    after = np.random.get_state()
    assert rng[0] == after[0] and np.array_equal(rng[1], after[1]) and rng[2:] == after[2:]
    for name in ("observations", "next_observations", "dones", "timeouts", "actions", "rewards"):
        np.testing.assert_array_equal(getattr(observed, name), getattr(plain, name))
    row = json.loads(path.read_text())
    assert row["transition_number"] == 1 and row["replay_position"] == 0
    assert row["done"] == row["timeout"] == 1 and row["effective_done"] == 0
    assert row["terminal_observation_sha256"] == row["stored_next_observation_sha256"]
    # An erroneous reset observation is retained as an observable mismatch,
    # never silently replaced or fixed by this observational subclass.
    observed.add(obs, obs, action, reward, done, infos)
    bad = json.loads(path.read_text().splitlines()[-1])
    assert bad["terminal_observation_sha256"] != bad["stored_next_observation_sha256"]


def test_untrained_official_cpu_checkpoint_roundtrip_preserves_all_parameters(tmp_path):
    pytest.importorskip("stable_baselines3")
    import torch
    from stable_baselines3 import TD3
    from scripts.btceth_rl_env import BTCETHPortfolioEnv
    assert not torch.cuda.is_initialized()
    td3.library_imports(td3.library_sources())
    env = BTCETHPortfolioEnv(dataset(), START, START + pd.Timedelta(days=33), training=True, episode_days=32)
    trace = tmp_path / "never_trained.jsonl"
    model = TD3("MlpPolicy", env, seed=1, device="cpu", **td3.model_parameters(trace))
    assert model.num_timesteps == model._n_updates == 0
    actual = td3.effective_parameters(model)
    for name in ("learning_rate", "gamma", "buffer_size", "learning_starts", "batch_size", "n_steps", "train_freq"):
        assert actual[name] == td3.recipe()[name]
    digest = original.parameter_digest(model.policy)
    model.save(tmp_path / "untrained.zip")
    reloaded = TD3.load(tmp_path / "untrained.zip", device="cpu")
    assert original.parameter_digest(reloaded.policy) == digest
    assert model.num_timesteps == reloaded.num_timesteps == 0
    assert not trace.exists()  # no learn, rollout, predict or replay.add
    assert not torch.cuda.is_initialized()
    env.close()


def test_validation_pass_label_alone_cannot_bypass_complete48_scope(tmp_path):
    td3.write_new(tmp_path / "independent_audit.validation.json", dict(passed=True, status="partial_test_pending",
        test_complete=False, training_model_count=10, validation_run_count=47, run_count=47,
        repeat_count=47, failure_count=0, failures=[]))
    with pytest.raises(ValueError, match="48-account"):
        td3.validation_audit(tmp_path, tmp_path)


def validation_rows(family, profile, ret=.1):
    return pd.DataFrame([dict(run_id=f"validation__{cost}__{family}__td3__{profile}__{key}",
        algorithm="td3", family=family, profile=profile, scenario=cost, split="validation",
        seed=None if key == "ensemble" else int(key), primary_policy=key == "ensemble", initial_capital=10000.,
        start=td3.VALIDATION[0], end_exclusive=td3.VALIDATION[1], net_return=ret, final_equity=10000 * (1 + ret),
        max_drawdown=-.1, fees=10., exposure_multiplier=td3.BUDGETS[family], rebalance_band=.025)
        for cost in td3.COSTS for key in (*map(str, td3.SEEDS), "ensemble")])


@pytest.fixture
def selection_case(tmp_path, monkeypatch):
    root = tmp_path / "new"; root.mkdir()
    td3.write_new(root / "protocol.json", dict(fixture=True))
    td3.write_new(root / "training_completion.json", dict(model_count=10))
    for profile in td3.PROFILES:
        for seed in td3.SEEDS:
            folder = root / "training" / td3.job_id(profile, seed); folder.mkdir(parents=True)
            (folder / "model.zip").write_bytes(b"fixture-only-model")
            (folder / "training_timeout_transitions.jsonl").write_text("fixture-only-not-a-run\n")
            td3.write_new(folder / "record.json", dict(algorithm="td3", profile=profile, seed=seed, actual_timesteps=65536,
                protocol_sha256=td3.sha(root / "protocol.json"), hyperparameters=td3.recipe(),
                model_relative_path=(folder / "model.zip").relative_to(root).as_posix(), model_sha256=td3.sha(folder / "model.zip"),
                timeout_trace_relative_path=(folder / "training_timeout_transitions.jsonl").relative_to(root).as_posix(),
                timeout_trace_sha256=td3.sha(folder / "training_timeout_transitions.jsonl")))
    for family in td3.BUDGETS:
        for profile in td3.PROFILES:
            folder = root / "validation" / family / profile; folder.mkdir(parents=True)
            validation_rows(family, profile).to_csv(folder / "summary.csv", index=False)
    for name in td3.validation_files():
        path = td3.relative(root, name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"fixture-only-account-evidence")
    td3.write_new(root / "independent_audit.validation.json", dict(fixture=True))
    monkeypatch.setattr(td3, "protocol", lambda *args: {})
    monkeypatch.setattr(td3, "terminal", lambda *args: {})
    monkeypatch.setattr(td3, "validation_audit", lambda *args: dict(input_sha256=td3.pins(root, td3.validation_files())))
    return root


def test_four_candidate_exact_ties_prefer_raw_then_profile_without_seed_picking(selection_case):
    root = selection_case
    value = td3.select(root.parent / "main", root.parent / "data", root)
    assert len(value["candidates"]) == 4
    assert (value["candidate"]["family"], value["candidate"]["profile"]) == ("raw", "balanced")
    assert value["selection_uses_test"] is False and value["seed_selection"] is False
    assert len(value["training_source_sha256"]) == 31
    assert td3.frozen_selection(root.parent / "main", root.parent / "data", root)[1] == value
    with pytest.raises(FileExistsError):
        td3.select(root.parent / "main", root.parent / "data", root)


def test_new_annual_marker_blocks_selection_even_if_other_results_would_look_better(selection_case):
    td3.write_new(selection_case / "test_started.json", {"at": "2026-10-30T00:00:00Z"})
    with pytest.raises(ValueError, match="precede"):
        td3.select(selection_case.parent / "main", selection_case.parent / "data", selection_case)


def test_changed_checkpoint_cannot_be_reused_for_selection(selection_case):
    path = selection_case / "training" / td3.job_id("return", 0) / "model.zip"
    path.write_bytes(b"changed-checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        td3.select(selection_case.parent / "main", selection_case.parent / "data", selection_case)


def test_resealed_but_fake_candidate_score_is_checked_against_actual_validation(selection_case):
    root = selection_case; td3.select(root.parent / "main", root.parent / "data", root)
    value = td3.read_json(root / "selection.json"); value["candidates"][0]["score"] += .01
    value["candidate"] = value["candidates"][0]; td3.publish(root / "selection.json", value)
    receipt = td3.read_json(root / "selection_receipt.json"); receipt["selection_sha256"] = td3.sha(root / "selection.json")
    td3.publish(root / "selection_receipt.json", receipt)
    with pytest.raises(ValueError, match="score"):
        td3.frozen_selection(root.parent / "main", root.parent / "data", root)


@pytest.mark.parametrize("damage", ["seed", "cost", "date", "capital", "algorithm"])
def test_candidate_population_and_funding_are_real_requirements(damage):
    frame = validation_rows("raw", "return")
    field, value = {"seed": ("seed", 99), "cost": ("scenario", "unknown"), "date": ("start", td3.TEST[0]),
        "capital": ("initial_capital", 5000.), "algorithm": ("algorithm", "sac")}[damage]
    frame.loc[0, field] = value
    with pytest.raises(ValueError):
        td3.candidate(frame, "raw", "return")


def test_annual_requires_separate_authorization_after_exact_seal(selection_case, monkeypatch):
    root = selection_case; td3.select(root.parent / "main", root.parent / "data", root)
    with pytest.raises(ValueError, match="Separate explicit"):
        td3.annual_gate(root.parent / "main", root.parent / "data", root, None, require_new=True)
    auth = root.parent / "authorization.json"
    td3.write_new(auth, dict(annual_evaluation_authorized=True, no_live_trading=True, financial_parameters_fixed=True,
        selection_sha256="wrong", authorized_at="2026-10-31T00:00:00Z", **{flag: True for flag in td3.FLAGS}))
    with pytest.raises(ValueError, match="identify"):
        td3.annual_gate(root.parent / "main", root.parent / "data", root, auth, require_new=True)


def test_terminal_proof_needs_actual_log_hash_and_exit_zero(tmp_path):
    root = tmp_path; td3.write_new(root / "protocol.json", {})
    (root / "fit.log").write_text("fixture completed\n")
    value = dict(phase="fit", status="completed", exit_code=0, protocol_sha256=td3.sha(root / "protocol.json"),
        supervisor_pid=1, child_pid=2,
        source_sha256=td3.sha(td3.__file__), log_relative_path="fit.log", log_sha256=td3.sha(root / "fit.log"))
    td3.write_new(root / "fit_process.json", value)
    assert td3.terminal(root, "fit") == value
    (root / "fit.log").write_text("altered terminal\n")
    with pytest.raises(ValueError, match="completed0"):
        td3.terminal(root, "fit")


def test_resource_allowance_cannot_hide_unknown_gpu_process_or_reused_pid():
    identity = dict(pid=10, uid=1000, username="fixture", exe="/usr/bin/ptyxis", starttime_ticks="123", boot_id="fixture")
    rows = [dict(gpu_uuid="GPU-fixture", pid=10)]
    allowance = dict(allowed_compute_instance=identity, allowed_gpu_rows=rows)
    td3.verify_resource_rows(rows, allowance, inspect_process=lambda pid: identity)
    with pytest.raises(RuntimeError):
        td3.verify_resource_rows(rows, allowance, inspect_process=lambda pid: {**identity, "starttime_ticks": "124"})
    with pytest.raises(RuntimeError):
        td3.verify_resource_rows([dict(gpu_uuid="GPU-fixture", pid=11)], allowance, inspect_process=lambda pid: identity)


@pytest.mark.parametrize("name", ["../outside", "C:/outside", "/outside"])
def test_evidence_path_escape_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="relative"):
        td3.relative(tmp_path, name)
