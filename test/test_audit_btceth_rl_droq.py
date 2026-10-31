"""Synthetic CPU-only independent audit checks, never market/model execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import audit_btceth_rl_droq as a


def test_exact_author_update_budget_and_random_action_boundary():
    counts = a.update_counts()
    assert counts == dict(environment_steps=32768, replay_adds=32768,
        random_exploration_actions=5001, learning_train_calls=27768,
        q_inner_rounds=555360, q_optimizer_steps=[555360, 555360],
        total_q_optimizer_steps=1110720, target_updates_per_network=[555360, 555360],
        policy_optimizer_steps=27768, alpha_optimizer_steps=27768)
    assert a.update_counts(5000)["q_inner_rounds"] == 0
    assert a.update_counts(5001)["q_inner_rounds"] == 20
    assert a.update_counts(5002)["random_exploration_actions"] == 5001


def test_actor_last_inner_round_updates_even_for_nonmultiple_delay():
    counts = a.update_counts(5001, utd_ratio=21)
    assert counts["q_inner_rounds"] == 21
    assert counts["policy_optimizer_steps"] == counts["alpha_optimizer_steps"] == 2
    assert a.update_counts(5001, auto_alpha=False)["alpha_optimizer_steps"] == 0


def test_hand_calculated_two_q_timeout_target_and_losses_do_not_consume_rng():
    state = np.random.get_state()
    target = a.q_target([1., 2.], [0., 1.], [[3., 5.], [100., 200.]], [-.5, -.1], alpha=.2, gamma=.99)
    np.testing.assert_allclose(target, [1 + .99 * 3.1, 2.])
    losses = a.q_loss_per_network([[1., 4.], [2., 1.]], [3., 2.])
    np.testing.assert_allclose(losses, [2., 1.])
    # The author's cat MSE times2 is a sum, not the mean of these two losses.
    assert losses.sum() == pytest.approx(3.)
    assert a.policy_loss([[1., 5.], [2., 4.]], [-1., -2.], .2) == pytest.approx(-3.3)
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), -1.])
def test_nonfinite_or_negative_temperature_rejected(alpha):
    with pytest.raises(ValueError):
        a.q_target([1.], [0.], [[1., 2.]], [-.1], alpha=alpha, gamma=.99)


def timeout_fixture():
    index = pd.date_range("2020-01-01", periods=66, tz="UTC")
    features = pd.DataFrame(np.arange(len(index) * 23).reshape(-1, 23) / 100., index=index)
    episodes, rows = [], []
    for i in (1, 2):
        end = index[i * 32]
        episodes.append(dict(completed_steps=32, end_exclusive=end.isoformat(), completed_until=end.isoformat()))
        row = dict(transition_number=i * 32, replay_position=i * 32 - 1, info_date=end.isoformat(),
            done=1., timeout=1., effective_done=0., stored_done=0., time_limit_truncated=True, env_index=0,
            terminated=False, truncated=True)
        for key, day in (("stored_observation", end - pd.Timedelta(days=1)),
                         ("terminal_observation", end), ("stored_next_observation", end)):
            value = np.r_[features.loc[day].to_numpy(), [0., 0., 1., 0.]].astype(np.float32)
            row[key + "_values"] = value.astype(float).tolist()
            row[key + "_sha256"] = a.observation_digest(value)
        rows.append(row)
    return rows, episodes, features


def test_timeout_actual_next_observation_and_no_future_feature_reads():
    rows, episodes, features = timeout_fixture()
    checks = a.Checks(); a.verify_timeout_trace(rows, episodes, features, checks, "toy", steps=64)
    assert checks.failed == 0
    # A future row may change; historical stored terminal evidence stays equal.
    features.iloc[-1] = 1000000.
    again = a.Checks(); a.verify_timeout_trace(rows, episodes, features, again, "toy", steps=64)
    assert again.failed == 0
    real = a.Checks(); a.verify_timeout_trace(rows, episodes, features, real, "real")
    assert real.failed > 0  # Two fixture traces cannot masquerade as1024.


@pytest.mark.parametrize("damage", ["reset", "absorbing", "future_prefix", "cutoff", "position"])
def test_false_timeout_or_future_inputs_fail(damage):
    rows, episodes, features = timeout_fixture()
    if damage == "reset":
        values = rows[0]["stored_next_observation_values"]
        values[-1] += .1; rows[0]["stored_next_observation_sha256"] = a.observation_digest(values)
    elif damage == "absorbing":
        rows[0]["stored_done"] = rows[0]["effective_done"] = 1.
    elif damage == "future_prefix":
        for key in ("terminal_observation", "stored_next_observation"):
            rows[0][key + "_values"][0] += 100
            rows[0][key + "_sha256"] = a.observation_digest(rows[0][key + "_values"])
    elif damage == "cutoff":
        rows[0]["info_date"] = episodes[0]["end_exclusive"] = episodes[0]["completed_until"] = "2025-01-02T00:00:00Z"
        features.loc[pd.Timestamp("2025-01-02", tz="UTC")] = features.iloc[32]
        features.loc[pd.Timestamp("2025-01-01", tz="UTC")] = features.iloc[31]
    else:
        rows[0]["replay_position"] += 1
    checks = a.Checks(); a.verify_timeout_trace(rows, episodes, features, checks, "toy", steps=64)
    assert checks.failed > 0


@pytest.fixture
def checkpoint(tmp_path):
    import torch
    tensors, groups, modules, trees = {}, {}, {}, {}
    policy = {"hidden_layers.0.weight": [256, 27], "hidden_layers.0.bias": [256],
        "hidden_layers.1.weight": [256, 256], "hidden_layers.1.bias": [256],
        "last_fc_layer.weight": [2, 256], "last_fc_layer.bias": [2],
        "last_fc_log_std.weight": [2, 256], "last_fc_log_std.bias": [2]}
    critic = {"hidden_layers.0.weight": [256, 29], "hidden_layers.0.bias": [256],
        "hidden_layers.2.weight": [256], "hidden_layers.2.bias": [256],
        "hidden_layers.3.weight": [256, 256], "hidden_layers.3.bias": [256],
        "hidden_layers.5.weight": [256], "hidden_layers.5.bias": [256],
        "last_fc_layer.weight": [1, 256], "last_fc_layer.bias": [1]}
    for group in a.PARAMETER_GROUPS:
        shapes = {"value": [1]} if group == "log_alpha" else policy if group == "policy" else critic
        names = {name: group + ":" + name for name in shapes}
        tensors.update({names[name]: torch.zeros(shape, dtype=torch.float32) for name, shape in shapes.items()})
        groups[group] = dict(tensor_keys=list(names.values()), named_tensors=names)
        if group != "log_alpha":
            modules[group] = dict(class_="", training=True, requires_grad_all=True,
                dropout_p=[] if group == "policy" else [.01, .01], layer_norm_shapes=[] if group == "policy" else [[256], [256]])
            modules[group]["class"] = "TanhGaussianPolicy" if group == "policy" else "Mlp"
            modules[group].pop("class_")
    counts = a.update_counts()
    for group, parameter_group, count in (("optimizer_policy", "policy", 27768), ("optimizer_q0", "q0", 555360),
                                         ("optimizer_q1", "q1", 555360), ("optimizer_alpha", "log_alpha", 27768)):
        keys = [group + ":step" + str(i) for i in range(len(groups[parameter_group]["tensor_keys"]))]
        tensors.update({key: torch.tensor(float(count)) for key in keys})
        all_keys, states = list(keys), []
        for index, (parameter, key) in enumerate(zip(groups[parameter_group]["tensor_keys"], keys)):
            moments = {}
            for kind in ("exp_avg", "exp_avg_sq"):
                moment_key = group + ":" + kind + str(index)
                tensors[moment_key] = torch.zeros_like(tensors[parameter]); all_keys.append(moment_key)
                moments[kind] = {"tensor": moment_key}
            states.append([index, {"dict": [["step", {"tensor": key}], *[[k, v] for k, v in moments.items()]]}])
        defaults = dict(params={"list": list(range(len(keys)))}, lr=.0003,
            betas={"tuple": [.9, .999]}, eps=1e-8, weight_decay=0., amsgrad=False)
        trees[group] = {"dict": [["state", {"dict": states}], ["param_groups", {"list": [{"dict": list(defaults.items())}]}]]}
        groups[group] = dict(tensor_keys=all_keys, step_tensor_keys=keys, step_values=[count] * len(keys))
    for group in ("rng_torch", "rng_numpy"):
        key = group + ":state"; tensors[key] = torch.zeros(8, dtype=torch.uint8)
        groups[group] = dict(tensor_keys=[key])
    names = {}
    for name, count in counts.items():
        if isinstance(count, int):
            names[name] = "counters:" + name; tensors[names[name]] = torch.tensor(count, dtype=torch.int64)
    groups["counters"] = dict(tensor_keys=list(names.values()), named_tensors=names)
    manifest = dict(tensor_groups=groups, modules=modules, trees=trees)
    tensor_file, manifest_file = tmp_path / "model_tensors.pt", tmp_path / "model.json"
    def save():
        manifest["tensors"] = {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in tensors.items()}
        torch.save(tensors, tensor_file)
        manifest.update(tensor_file=tensor_file.name, tensor_sha256=a.sha(tensor_file), upstream_commit=a.UPSTREAM_COMMIT,
            upstream_manifest_sha256=a.UPSTREAM_MANIFEST_SHA256, effective_target_entropy=-2, hidden_sizes=[256, 256], action_limit=1.,
            counters=counts, optimizer_classes={key: "Adam" for key in a.OPTIMIZER_GROUPS},
            log_alpha_value=0., effective_alpha=1., parameter_sha256=a.tensor_parameter_digest(tensors, groups))
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    save()
    return tensors, manifest, tensor_file, manifest_file, save


def test_safe_cpu_tensor_checkpoint_counts_and_rng_are_independent(checkpoint):
    import torch
    _, _, tensor_file, manifest_file, _ = checkpoint
    rng = torch.get_rng_state().clone(); cuda = torch.cuda.is_initialized()
    result = a.read_tensor_checkpoint(tensor_file, manifest_file)
    assert result["actual_counters"]["q_inner_rounds"] == 555360
    assert len(result["parameter_sha256"]) == 64
    assert result["cpu_rng_consumed"] is False and result["cuda_initialized_changed"] is False
    assert torch.equal(rng, torch.get_rng_state()) and cuda == torch.cuda.is_initialized()


@pytest.mark.parametrize("damage", ["optimizer_step", "missing_ln", "target_eval", "dropout", "nan", "steps", "unknown_tensor", "parameter_dtype"])
def test_actual_tensor_or_target_mode_tampering_is_detected(checkpoint, damage):
    import torch
    tensors, manifest, tensor_file, manifest_file, save = checkpoint
    if damage == "optimizer_step":
        tensors[manifest["tensor_groups"]["optimizer_q0"]["step_tensor_keys"][0]] -= 1
    elif damage == "missing_ln":
        group = manifest["tensor_groups"]["optimizer_q0"]
        group["step_tensor_keys"] = group["step_tensor_keys"][:-1]; group["step_values"] = group["step_values"][:-1]
    elif damage == "target_eval":
        manifest["modules"]["target_q0"]["training"] = False
    elif damage == "dropout":
        manifest["modules"]["q0"]["dropout_p"] = [0., 0.]
    elif damage == "nan":
        tensors[manifest["tensor_groups"]["policy"]["tensor_keys"][0]][0, 0] = float("nan")
    elif damage == "steps":
        tensors[manifest["tensor_groups"]["counters"]["named_tensors"]["environment_steps"]] += 1
    elif damage == "parameter_dtype":
        key = manifest["tensor_groups"]["policy"]["tensor_keys"][0]
        tensors[key] = tensors[key].to(torch.int64)
    else:
        tensors["unclaimed"] = torch.tensor(0.)
    save()
    with pytest.raises(ValueError):
        a.read_tensor_checkpoint(tensor_file, manifest_file)


def test_parameter_digest_does_not_include_optimizer_or_rng(checkpoint):
    tensors, manifest, _, _, _ = checkpoint
    old = a.tensor_parameter_digest(tensors, manifest["tensor_groups"])
    tensors[manifest["tensor_groups"]["rng_torch"]["tensor_keys"][0]][0] = 1
    tensors[manifest["tensor_groups"]["optimizer_policy"]["step_tensor_keys"][0]] += 1
    assert a.tensor_parameter_digest(tensors, manifest["tensor_groups"]) == old


def test_original_author_cpu_one_update_checkpoint_is_independently_readable(tmp_path):
    """One synthetic CPU math update, no market environment or GPU operation."""
    import torch
    from scripts import btceth_rl_droq as runner
    numpy_rng, threads, bytecode = np.random.get_state(), torch.get_num_threads(), sys.dont_write_bytecode
    cuda_before = torch.cuda.is_initialized()
    try:
        torch.set_num_threads(1)
        sys.dont_write_bytecode = True
        with torch.random.fork_rng(devices=[]):
            runner.seed_cpu(5)
            agent = runner.make_agent(overrides=dict(start_steps=0, delay_update_steps=0, utd_ratio=1,
                policy_update_delay=1, batch_size=2, replay_size=8))
            runner.observe_agent(agent)
            agent.store_data(np.zeros(27, dtype=np.float32), np.zeros(2, dtype=np.float32),
                             .1, np.ones(27, dtype=np.float32), False)
            runner.original_train(agent, runner.MetricSink())
            counters = runner.actual_counters(agent, environment_steps=1, random_actions=1)
            expected = a.update_counts(1, start_steps=0, delay_update_steps=0, utd_ratio=1, policy_update_delay=1)
            assert counters == expected
            directory = tmp_path / "cpu_math_checkpoint"
            manifest = runner.save_checkpoint(directory, agent, counters=counters)
            result = a.read_tensor_checkpoint(directory / "model_tensors.pt", directory / "model.json", expected_counters=expected)
            assert result["parameter_sha256"] == manifest["parameter_sha256"] == runner.parameter_digest(agent)
            assert cuda_before == torch.cuda.is_initialized()
    finally:
        np.random.set_state(numpy_rng); torch.set_num_threads(threads); sys.dont_write_bytecode = bytecode


def summary_fixture(family, *, base=.1, stress=.05):
    rows = []
    for cost, value in (("base", base), ("stress", stress)):
        for key in (*a.SEEDS, "ensemble"):
            rows.append(dict(run_id=f"validation__{cost}__{family}__droq__balanced__{key}",
                algorithm="droq", family=family, profile="balanced", scenario=cost, split="validation",
                seed=float("nan") if key == "ensemble" else key, primary_policy=key == "ensemble",
                initial_capital=10000., start=a.VALIDATION[0], end_exclusive=a.VALIDATION[1],
                net_return=value, max_drawdown=-.1, final_equity=10000 * (1 + value), fees=10.,
                rebalance_band=.025, exposure_multiplier=a.FAMILIES[family]))
    return pd.DataFrame(rows)


def test_only_two_budget_candidates_all_five_seeds_and_raw_first_ties():
    groups = {(family, "balanced"): summary_fixture(family) for family in a.FAMILIES}
    ranked = a.rank_candidates(groups)
    assert len(ranked) == 2 and [row["family"] for row in ranked] == ["raw", "cap075"]
    missing = {key: value.copy() for key, value in groups.items()}
    missing[("raw", "balanced")] = missing[("raw", "balanced")].iloc[:-1]
    with pytest.raises(ValueError):
        a.rank_candidates(missing)
    changed_band = groups[("raw", "balanced")].copy(); changed_band["rebalance_band"] = .05
    with pytest.raises(ValueError):
        a.validate_group_rows(changed_band, split="validation", family="raw", profile="balanced")


def test_annual_phase_fails_closed_before_any_price_or_tensor_read(tmp_path, monkeypatch):
    root, main = tmp_path / "new", tmp_path / "old"; root.mkdir(); main.mkdir()
    dataset = tmp_path / "unparsed_dataset"; dataset.write_bytes(b"NOT_MARKET_DATA")
    def forbidden(*args, **kwargs):
        pytest.fail("No source, market or checkpoint may be parsed before a sealed annual authorization")
    monkeypatch.setattr(a, "verify_protocol", forbidden)
    monkeypatch.setattr(a, "load_market", forbidden)
    result = a.audit(root, main, dataset, phase="test")
    assert result["passed"] is False and result["status"] == "failed" and result["test_complete"] is False
    assert result["test_run_count"] == result["training_model_count"] == 0


def test_original_validation_audit_refuses_later_year_marker_before_inputs(tmp_path, monkeypatch):
    root, main = tmp_path / "new", tmp_path / "old"; root.mkdir(); main.mkdir()
    (root / "test_started.json").write_bytes(b"NOT_PARSED_YEAR_MARKER")
    dataset = tmp_path / "unparsed_dataset"; dataset.write_bytes(b"NOT_MARKET_DATA")
    monkeypatch.setattr(a, "verify_protocol", lambda *args: pytest.fail("Year marker must block original validation before reads"))
    result = a.audit(root, main, dataset)
    assert result["passed"] is False and result["validation_run_count"] == 0


def test_optimizer_tagged_tree_cannot_point_step_mirror_at_another_tensor(checkpoint):
    tensors, manifest, tensor_file, manifest_file, save = checkpoint
    tree = manifest["trees"]["optimizer_q0"]
    state = tree["dict"][0][1]["dict"][0][1]["dict"]
    state[0][1] = {"tensor": manifest["tensor_groups"]["optimizer_q0"]["tensor_keys"][-1]}
    save()
    with pytest.raises(ValueError, match="step mirrors"):
        a.read_tensor_checkpoint(tensor_file, manifest_file)


@pytest.mark.parametrize("which", ["study", "main", "source"])
def test_original_root_links_rejected_before_resolution(tmp_path, monkeypatch, which):
    paths = {name: tmp_path / name for name in ("study", "main", "source")}
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == paths[which] or original(path))
    with pytest.raises(ValueError, match="links or junctions"):
        a.Inputs(paths["study"], paths["main"], paths["source"], a.Checks())
    assert not paths[which].exists()


def fit_authorization_fixture(tmp_path):
    root, main, source = (tmp_path / name for name in ("study", "main", "source"))
    path = root / "preflight/fit_validation_authorization.json"; path.parent.mkdir(parents=True)
    sources = {name: "a" * 64 for name in a.SOURCE_FILES}
    approval = dict(fit_validation_authorized=True, no_live_trading=True, reviewed_source_sha256=sources,
        hyperparameters=deepcopy(a.RECIPE), seeds=list(a.SEEDS), timesteps_per_model=32768, budgets=dict(a.FAMILIES),
        authorized_at="2026-10-30T01:00:00Z", **{flag: True for flag in a.FLAGS})
    path.write_text(json.dumps(approval))
    spec = dict(source_sha256=sources, fit_validation_authorization_sha256=a.sha(path), prepared_at="2026-10-30T01:01:00Z")
    checks = a.Checks(); inputs = a.Inputs(root, main, source, checks)
    return path, approval, spec, inputs, checks


def test_fit_validation_authorization_is_independently_pinned(tmp_path):
    path, _, spec, inputs, checks = fit_authorization_fixture(tmp_path)
    a.verify_fit_authorization(inputs, spec)
    assert checks.failed == 0
    assert inputs.hashes["study"] == {"preflight/fit_validation_authorization.json": a.sha(path)}


@pytest.mark.parametrize("damage", ["source", "recipe", "seed", "steps", "budgets", "late", "sha", "yearflag"])
def test_fit_authorization_scope_or_timing_tamper_is_rejected(tmp_path, damage):
    path, approval, spec, inputs, checks = fit_authorization_fixture(tmp_path)
    if damage == "source":
        approval["reviewed_source_sha256"] = dict(approval["reviewed_source_sha256"])
        approval["reviewed_source_sha256"][next(iter(a.SOURCE_FILES))] = "b" * 64
    elif damage == "recipe":
        approval["hyperparameters"]["gamma"] = .995
    elif damage == "seed":
        approval["seeds"] = [0, 1]
    elif damage == "steps":
        approval["timesteps_per_model"] = 65536
    elif damage == "budgets":
        approval["budgets"] = {"raw": 1.}
    elif damage == "late":
        approval["authorized_at"] = "2026-10-30T01:02:00Z"
    elif damage == "yearflag":
        approval[a.FLAGS[0]] = False
    path.write_text(json.dumps(approval))
    if damage != "sha":
        # Even a matching whole-file SHA cannot replace the reviewed scope.
        spec["fit_validation_authorization_sha256"] = a.sha(path)
    else:
        spec["fit_validation_authorization_sha256"] = "0" * 64
    a.verify_fit_authorization(inputs, spec)
    assert checks.failed > 0


def gpu_fixture():
    return [dict(index=i, uuid="GPU-" + str(i), name="Tesla V100", memory_total_mib=16384,
                 memory_used_mib=19, utilization_percent=0) for i in (0, 1)]


def test_dynamic_gpu_memory_and_utilization_are_real_observations_not_identity():
    declared, actual = gpu_fixture(), gpu_fixture()
    actual[0].update(memory_used_mib=431, utilization_percent=2)
    checks = a.Checks(); a.verify_gpu_snapshot(actual, declared, checks, "phase")
    assert checks.failed == 0


@pytest.mark.parametrize("field,value", [("uuid", "GPU-other"), ("memory_total_mib", 32768),
                                         ("memory_used_mib", -1), ("utilization_percent", float("nan"))])
def test_gpu_identity_change_or_invalid_dynamic_observation_is_rejected(field, value):
    declared, actual = gpu_fixture(), gpu_fixture(); actual[0][field] = value
    checks = a.Checks(); a.verify_gpu_snapshot(actual, declared, checks, "phase")
    assert checks.failed > 0


def test_independent_action_space_pcg64_rebuild_matches_true_gym_box():
    from gymnasium.spaces import Box
    box = Box(-1., 1., shape=(2,), dtype=np.float32); box.seed(7)
    before = deepcopy(box.np_random.bit_generator.state)
    for _ in range(11):
        box.sample()
    after = deepcopy(box.np_random.bit_generator.state)
    global_before = np.random.get_state()
    checks = a.Checks(); a.verify_action_space_rng(before, after, 7, 11, checks, "toy")
    assert checks.failed == 0
    global_after = np.random.get_state()
    assert np.array_equal(global_before[1], global_after[1]) and global_before[2:] == global_after[2:]
    for wrong_seed, wrong_draws in ((8, 11), (7, 12)):
        damaged = a.Checks(); a.verify_action_space_rng(before, after, wrong_seed, wrong_draws, damaged, "toy")
        assert damaged.failed > 0
