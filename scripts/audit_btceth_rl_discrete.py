"""Independent primitives for the new discrete BTC/ETH research protocol.

No training, environment construction, policy loading or market loading occurs
here. Safe weights-only reads inspect inert SB3 state dictionaries; the legacy
independent cash/inventory arithmetic is reused, never the execution ledger.
Full protocol gates will be added only after the new contract is declared.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

from scripts import audit_btceth_rl as financial
from scripts import audit_btceth_rl_exposure_cap as inventory

ALGORITHMS = ("qrdqn", "dqn")
VARIANTS = ("qrdqn_mean", "qrdqn_lower_tail25", "dqn_mean")
BUDGETS = {"raw": 1., "cap075": .75}
SEEDS = tuple(range(5))
PROFILE = "balanced"
STEPS, EPISODE_DAYS, OBSERVATIONS, ACTIONS, QUANTILES = 32768, 32, 27, 21, 200
EXPECTED_COUNTS = dict(environment_steps=32768, replay_adds=32768, optimizer_steps=7936,
    target_sync_calls=64, completed_episodes=1024, unused_zero_step_resets=1,
    random_warmup_actions=1024)
TARGETS = dict(primary_base_net_return_at_least=.05, primary_base_max_drawdown_magnitude_at_most=.20,
    primary_stress_net_return_at_least=0., primary_positive_training_seeds_at_least=3,
    primary_beats_equal_hold_return_and_drawdown=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def utc(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    require(stamp.tzinfo is not None, "Explicit timezone required")
    return stamp.astimezone(timezone.utc)


def matched_recipe(algorithm):
    require(algorithm in ALGORITHMS, "Only the two declared native implementations")
    values = dict(learning_rate=1e-4, buffer_size=50000, learning_starts=1024, batch_size=128,
        gamma=.99, tau=1., train_freq=[4, "step"], gradient_steps=1, n_steps=1,
        target_update_interval=512, exploration_initial_eps=1., exploration_final_eps=.05,
        exploration_fraction=.25, max_grad_norm=10., optimize_memory_usage=False,
        replay_buffer_kwargs=dict(handle_timeout_termination=True),
        policy_kwargs=dict(net_arch=[128, 128], activation_fn="ReLU", optimizer_class="Adam",
                           optimizer_kwargs=dict(eps=1e-8)))
    if algorithm == "qrdqn":
        values["policy_kwargs"]["n_quantiles"] = QUANTILES
    return values


def expected_update_counts(steps=STEPS):
    require(is_int(steps) and steps == STEPS, "This finite recipe fixes32768 interactions")
    # SB3 trains only after a4-step rollout with num_timesteps > learning_starts.
    updates = sum(step > 1024 for step in range(4, steps + 1, 4))
    return {**EXPECTED_COUNTS, "optimizer_steps": updates, "target_sync_calls": steps // 512}


def validate_observed_counts(values):
    require(isinstance(values, dict), "Actual counter dictionary required")
    require(all(type(values.get(key)) is int and values[key] == expected for key, expected in expected_update_counts().items()),
            "Actual finite interactions, optimizer steps, syncs, episodes and unused reset must all match")


def action_to_box(action_id):
    require(is_int(action_id) and 0 <= action_id < ACTIONS, "One integer action ID0..20 required; no float/bool/clip")
    if action_id == 0:
        return np.array([-1., 0.])
    gross_index, share_index = divmod(int(action_id) - 1, 5)
    return np.array([(.25, .5, .75, 1.)[gross_index], 2 * (0., .25, .5, .75, 1.)[share_index] - 1.])


def action_target(action_id, family="raw"):
    require(family in BUDGETS, "Exactly raw or the declared .75 overlay")
    action = action_to_box(action_id)
    gross, eth_share = .95 * max(float(action[0]), 0.), (float(action[1]) + 1.) / 2
    coins = np.array([gross * (1 - eth_share), gross * eth_share]) * BUDGETS[family]
    return np.r_[coins, 1 - float(coins.sum())]


def ensemble_target(action_ids, family="raw"):
    require(len(action_ids) == 5, "All five fixed model actions, never chosen seeds")
    # This is a target-weight mean. Averaging categorical IDs/Box coordinates
    # is invalid and would change the existing common funded-account semantics.
    return np.mean([action_target(action_id, family) for action_id in action_ids], axis=0)


def inference_scores(values, variant):
    require(variant in VARIANTS, "Exactly mean/lower-tail25 QR or mean DQN")
    values = np.asarray(values, dtype=np.float64)
    require(np.isfinite(values).all(), "All predicted action values must be finite")
    if variant == "dqn_mean":
        require(values.ndim in (1, 2) and values.shape[-1] == ACTIONS, "DQN has21 scalar Q values")
        return values.copy()
    require(values.ndim in (2, 3) and values.shape[-2:] == (QUANTILES, ACTIONS), "QR has200 fixed tau heads by21 actions")
    # Preserve nominal tau labels. Never sort outputs into a different policy.
    return values[..., :50, :].mean(axis=-2) if variant == "qrdqn_lower_tail25" else values.mean(axis=-2)


def greedy_action(values, variant):
    return np.argmax(inference_scores(values, variant), axis=-1)


def crossing_diagnostics(quantiles):
    values = np.asarray(quantiles, dtype=float)
    require(values.ndim in (2, 3) and values.shape[-2:] == (200, 21) and np.isfinite(values).all(), "Finite complete fixed quantile heads required")
    inversions = np.diff(values, axis=-2) < 0
    return dict(adjacent_inversion_count=int(inversions.sum()), adjacent_pair_count=int(inversions.size),
        adjacent_crossing_fraction=float(inversions.mean()), low_tail_boundary_crossing_count=int((values[..., 49, :] > values[..., 50, :]).sum()),
        sorted_outputs_used=False, nominal_tau_min=.0025, nominal_tau_max_lower_tail=.2475,
        cvar_optimality_guarantee=False)


def verify_inference_trace(values, variant, recorded_scores, action_id, *, atol=2e-5):
    expected = inference_scores(values, variant)
    actual = np.asarray(recorded_scores, dtype=float)
    require(expected.shape == actual.shape == (21,) and np.isfinite(actual).all(), "Full21 scores per original observation required")
    require(np.allclose(actual, expected, rtol=1e-6, atol=atol), "Recorded scores must match independent fixed mean/tau-head arithmetic")
    require(is_int(action_id) and int(np.argmax(actual)) == action_id, "Actual float32 score argmax with first-ID exact ties required")
    # Different CUDA/NumPy reduction trees can perturb near-ties; prove that
    # the chosen action remains within the declared arithmetic tolerance.
    require(expected.max() - expected[action_id] <= atol + 1e-6 * abs(expected.max()), "Selected action is not a fixed-score maximizer")


def observation_digest(values):
    values = np.asarray(values, dtype="<f4")
    require(values.shape == (27,) and np.isfinite(values).all(), "Full finite27-dimensional causal state required")
    return hashlib.sha256(values.tobytes()).hexdigest()


def extend_transition_chain(previous_digest, observation, next_observation, action_id, reward, done, timeout):
    require(isinstance(previous_digest, str) and len(previous_digest) == 64 and set(previous_digest) <= set("0123456789abcdef"), "Previous observer chain SHA required")
    action_to_box(action_id)
    obs, nxt = np.asarray(observation, dtype="<f4"), np.asarray(next_observation, dtype="<f4")
    observation_digest(obs); observation_digest(nxt)
    values = np.asarray([reward, done, timeout], dtype="<f4")
    require(np.isfinite(values).all() and done in (0., 1.) and timeout in (0., 1.), "Actual finite reward and binary stored flags")
    data = bytes.fromhex(previous_digest) + obs.tobytes() + nxt.tobytes() + np.asarray([action_id], dtype="<i8").tobytes() + values.tobytes()
    return hashlib.sha256(data).hexdigest()


def audit_timeout_rows(rows, *, expected_episodes=1024):
    """Inspect actual post-super.add arrays, never substitute a reset state."""
    require(len(rows) == expected_episodes, "Every32-day timeout must be logged")
    chains = set()
    for episode, row in enumerate(rows, 1):
        step = episode * 32
        require(type(row.get("transition_number")) is int and row["transition_number"] == step
                and row.get("replay_position") == step - 1 and row.get("env_index") == 0, "Exact actual nonwrapping replay position")
        require(row.get("stored_done") == 1. and row.get("stored_timeout") == 1.
                and row.get("effective_done_from_stored_flags") == 0. and row.get("TimeLimit.truncated") is True
                and row.get("terminated_from_stored_flags") is False, "Truncation retains rawdone but bootstraps economically")
        require(row.get("handle_timeout_termination") is True and row.get("terminal_observation_present") is True
                and row.get("stored_next_matches_terminal") is True, "Native timeout mask and true terminal state remain enabled")
        require(is_int(row.get("action_id")) and 0 <= row["action_id"] < 21, "Replay stores original categorical action, never mapped Box coordinates")
        chain = row.get("transition_chain_sha256", "")
        require(isinstance(chain, str) and len(chain) == 64 and set(chain) <= set("0123456789abcdef"), "Actual observer transition-chain identity required")
        require(chain not in chains, "Every timeout must come from a distinct actual chain position")
        chains.add(chain)
        actual = {}
        for key in ("stored_observation", "terminal_observation", "stored_next_observation"):
            values = np.asarray(row.get(key + "_values"), dtype="<f4")
            require(values.shape == (27,) and np.isfinite(values).all() and observation_digest(values) == row.get(key + "_sha256"), "Actual full replay/terminal array and SHA required")
            actual[key] = values
        require(np.array_equal(actual["terminal_observation"], actual["stored_next_observation"]), "Replay next observation must be true terminal state, not reset state")
    return dict(timeout_rows=len(rows), all_terminal_next_observations_verified=True, native_timeout_mask=True,
        final_transition_chain_sha256=rows[-1]["transition_chain_sha256"] if rows else None)


def verify_training_history(history, *, eligible_starts=None, train=("2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z")):
    require(len(history) == 1025, "1024 completed32-day fragments plus the declared unused zero-step auto-reset")
    first, last = map(utc, train)
    eligible = None if eligible_starts is None else {utc(value) for value in eligible_starts}
    for index, item in enumerate(history):
        start, end, until = (utc(item[key]) for key in ("start", "end_exclusive", "completed_until"))
        require(first <= start < end <= last and end - start == timedelta(days=32) and (eligible is None or start in eligible), "Every sampled fragment must be a complete causal eligible TRAIN window")
        if index < 1024:
            require(until == end and item.get("completed_steps") == 32 and item.get("terminated") is False and item.get("truncated") is True, "Every budgeted episode really completed exactly32 truncated days")
        else:
            require(until == start and item.get("completed_steps") == 0 and item.get("terminated") is False and item.get("truncated") is False, "Only the final unused zero-step reset is excluded from trained transitions")
    return dict(completed_episodes=1024, completed_steps=32768, unused_zero_step_resets=1)


def expected_policy_shapes(algorithm):
    require(algorithm in ALGORITHMS, "Declared DQN or QR-DQN weights only")
    name = "quantile_net" if algorithm == "qrdqn" else "q_net"
    output = 200 * 21 if algorithm == "qrdqn" else 21
    shapes = {"0.weight": (128, 27), "0.bias": (128,), "2.weight": (128, 128), "2.bias": (128,), "4.weight": (output, 128), "4.bias": (output,)}
    return {prefix + "." + name + "." + suffix: size for prefix in (name, name + "_target") for suffix, size in shapes.items()}


def inspect_state_dicts(policy, optimizer, data, algorithm):
    """Safe inert tensors only; no SB3 .load(), unpickling objects or NN creation."""
    import torch
    shapes = expected_policy_shapes(algorithm)
    require(isinstance(policy, dict) and set(policy) == set(shapes), "Exact online+target network key inventory required")
    for name, size in shapes.items():
        tensor = policy[name]
        require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == size and tensor.dtype == torch.float32 and torch.isfinite(tensor).all().item(), "Exact finite FP32 matched-recipe parameter shape: " + name)
    require(data.get("num_timesteps") == 32768 and type(data.get("num_timesteps")) is int
            and data.get("_n_updates") == 7936 and type(data.get("_n_updates")) is int
            and data.get("_n_calls") == 32768 and type(data.get("_n_calls")) is int, "Real SB3 interaction, on-step and gradient update counters required")
    require(isinstance(optimizer, dict) and set(optimizer) == {"state", "param_groups"} and len(optimizer["param_groups"]) == 1, "One actual native Adam state dictionary required")
    group = optimizer["param_groups"][0]; ids = group["params"]
    require(len(ids) == len(set(ids)) == 6 and all(type(i) is int for i in ids) and set(optimizer["state"]) == set(ids), "All six online tensors, no target tensors, have actual optimizer state")
    require(group.get("lr") == 1e-4 and group.get("eps") == 1e-8 and tuple(group.get("betas", [])) == (.9, .999)
            and group.get("weight_decay") == 0 and group.get("amsgrad") is False, "Actual matched Adam defaults/class/explicit epsilon required")
    prefix = "quantile_net" if algorithm == "qrdqn" else "q_net"
    online = [name for name in shapes if name.startswith(prefix + ".")]
    steps = []
    for identity, name in zip(ids, online):
        entry = optimizer["state"][identity]
        require(set(entry) == {"step", "exp_avg", "exp_avg_sq"}, "Native Adam moments must belong to every actual online parameter")
        step = entry["step"]
        require(isinstance(step, torch.Tensor) and step.numel() == 1 and float(step.item()) == 7936., "Real optimizer tensor step, not a declared JSON count")
        for key in ("exp_avg", "exp_avg_sq"):
            value = entry[key]
            require(isinstance(value, torch.Tensor) and tuple(value.shape) == shapes[name] and value.dtype == torch.float32 and torch.isfinite(value).all().item(), "Actual finite correctly shaped Adam moment")
        require((entry["exp_avg_sq"] >= 0).all().item(), "Adam squared moment cannot be negative")
        steps.append(int(step.item()))
    return dict(algorithm=algorithm, policy_tensor_count=12, online_optimizer_tensor_count=6,
        actual_optimizer_steps=steps, policy_shapes={k: list(v) for k, v in shapes.items()}, tensor_payload_only=True)


def inspect_sb3_zip(path, algorithm):
    import torch
    path = Path(path)
    require(not path.is_symlink(), "Model source cannot be a link")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist(); names = [member.filename for member in members]
        required = {"data", "policy.pth", "policy.optimizer.pth", "pytorch_variables.pth", "_stable_baselines3_version", "system_info.txt"}
        require(len(names) == len(set(names)) and set(names) == required, "Exact ordinary SB3 archive entries required")
        for member in members:
            key = PurePosixPath(member.filename)
            require(not key.is_absolute() and ".." not in key.parts and not member.is_dir() and member.file_size <= 128 * 1024 * 1024
                    and (member.external_attr >> 16) & 0o170000 != 0o120000, "No traversal/link/oversized zip payload")
        require(archive.read("_stable_baselines3_version").decode().strip() == "2.9.0", "Frozen actual SB32.9.0 serialization required")
        data = json.loads(archive.read("data"))  # cloudpickle/base64 fields remain inert text
        policy = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True)
        optimizer = torch.load(io.BytesIO(archive.read("policy.optimizer.pth")), map_location="cpu", weights_only=True)
        variables = torch.load(io.BytesIO(archive.read("pytorch_variables.pth")), map_location="cpu", weights_only=True)
        require(isinstance(variables, dict) and not variables, "No hidden actor/alpha or alternate serialized trainable state")
        value = inspect_state_dicts(policy, optimizer, data, algorithm)
    value["model_sha256"] = financial.sha(path)
    return value


def fixed_score(base, stress):
    for row in (base, stress):
        require(all(math.isfinite(float(row[key])) for key in ("net_return", "max_drawdown")) and -1 < float(row["net_return"]) and -1 < float(row["max_drawdown"]) <= 0, "Finite funded validation metrics required")
    return financial.score_candidate(base, stress)


def rank_candidate_pool(candidates):
    require(len(candidates) == 6 and {(v["variant"], v["family"]) for v in candidates} == {(v, family) for v in VARIANTS for family in BUDGETS}, "Exactly the six predeclared inference/budget candidates")
    rows = []
    for row in candidates:
        require(row.get("profile") == "balanced", "One fixed shaped reward profile")
        truth = fixed_score(row["base"], row["stress"])
        if "score" in row:
            require(math.isclose(float(row["score"]), truth, rel_tol=1e-12, abs_tol=1e-12), "Original fixed validation score only")
        rows.append({**row, "score": truth})
    ordered = sorted(rows, key=lambda v: (-v["score"], 0 if v["family"] == "raw" else 1, v["variant"], v["profile"]))
    return dict(candidates=ordered, global_primary=ordered[0], per_variant={variant: next(v for v in ordered if v["variant"] == variant) for variant in VARIANTS}, selection_uses_test=False)


def case_population(split, per_variant=None):
    require(split in ("validation", "test"), "Only the two frozen evaluation windows")
    if split == "validation":
        require(per_variant is None, "Validation cannot choose only one budget")
        pairs = [(variant, family) for variant in VARIANTS for family in BUDGETS]
    else:
        require(isinstance(per_variant, dict) and set(per_variant) == set(VARIANTS), "Annual discloses all three variants at their own presealed budget")
        require(all(value in BUDGETS for value in per_variant.values()), "Only predeclared raw/cap075 annual budgets")
        pairs = [(variant, per_variant[variant]) for variant in VARIANTS]
    return [dict(variant=variant, family=family, scenario=scenario, key=key)
            for variant, family in pairs for scenario in ("base", "stress") for key in (*map(str, SEEDS), "ensemble")]


def primary_targets(base, stress, base_seed_returns, equal_hold):
    require(len(base_seed_returns) == 5 and {v["seed"] for v in base_seed_returns} == set(SEEDS), "All five original training seeds, never a subset")
    require(all(is_int(v["seed"]) for v in base_seed_returns), "Original integer seed identities required")
    require(all(math.isfinite(float(v["net_return"])) for v in [base, stress, equal_hold, *base_seed_returns]), "No missing returns filled with cash0")
    require(all(math.isfinite(float(v["max_drawdown"])) and -1 < float(v["max_drawdown"]) <= 0 for v in [base, stress, equal_hold]), "Finite actual funded hourly drawdowns required")
    positive = sum(float(v["net_return"]) > 0 for v in base_seed_returns)
    gates = dict(base_return_at_least_5pct=float(base["net_return"]) >= .05,
        base_max_drawdown_at_most_20pct=abs(float(base["max_drawdown"])) <= .20,
        stress_return_nonnegative=float(stress["net_return"]) >= 0,
        at_least_three_positive_training_seeds=positive >= 3,
        beats_equal_hold_return_and_drawdown=float(base["net_return"]) > float(equal_hold["net_return"]) and float(base["max_drawdown"]) > float(equal_hold["max_drawdown"]))
    return dict(gates=gates, goal_performance_achieved=all(gates.values()), positive_base_training_seeds=positive,
        base_seed_returns=base_seed_returns, fresh_holdout_success=False, accounting_passage_is_not_goal_success=True)


def reconstruct_account(directory, dataset, *, start, end, fee, slippage, checks, label):
    """Reuse the original independent price/cash math plus exact SELL-dust inventory."""
    directory = Path(directory)
    equity, fills = financial.read_equity(directory / "equity_1h.csv.gz"), financial.read_csv(directory / "fills.csv")
    metadata = financial.read_json(directory / "metadata.json")
    rebuilt = financial.reconstruct_fills(dataset, fills, equity, metadata, start=start, end=end, fee=fee, slippage=slippage, checks=checks, label=label)
    closing = dataset["hourly_close"].loc[rebuilt["equity"].index, ["BTCUSDT", "ETHUSDT"]]
    inventory.attach_inventory_replay(rebuilt, fills, closing, checks=checks, label=label, metadata=metadata)
    return rebuilt
