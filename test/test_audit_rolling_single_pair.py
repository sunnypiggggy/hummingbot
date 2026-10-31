"""Meaningful leakage, training-budget and account-continuity gate cases."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import hashlib
import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts import audit_rolling_single_pair as audit


def snapshot():
    return dict(cash=5000., positions={"BTCUSDT": .05}, cost_basis={"BTCUSDT": 4000.},
                realized_pnl=1000., nav=11000., high_water=12000., fill_count=2,
                fills_digest="a" * 64, equity_rows=24, weight_rows=1, boundary_rows=2,
                fees=9., notional=9000., slippage_cost=1.8, steps=1,
                last_prices={"BTCUSDT": 120000.}, date="2025-10-24T00:00:00Z")


def recipe(algorithm="sac"):
    values = dict(policy="MlpPolicy", learning_rate=.0003, gamma=.995,
                  buffer_size=100000, learning_starts=512, batch_size=128,
                  tau=.005, train_freq=1, gradient_steps=1,
                  policy_kwargs={"net_arch": [128, 128]})
    if algorithm == "td3":
        values.update(train_freq=[1, "step"], policy_delay=2, n_steps=1,
                      target_policy_noise=.2, target_noise_clip=.5)
    else:
        values["ent_coef"] = "auto"
    return values


def model_fixture(tmp_path, algorithm="sac"):
    context = audit.model_context(audit.schedule_versions()[0], symbol="BTCUSDT", algorithm=algorithm,
                                  recipe_id=algorithm + "_default", seed=0, protocol_sha256="b" * 64)
    native_recipe = recipe(algorithm)
    steps = 65536 if algorithm == "td3" else 32768
    updates = steps - 512
    start = audit.utc(context["train_start"])
    ep = dict(start=audit.stamp(start), end_exclusive=audit.stamp(start + timedelta(days=32)),
              completed_until=audit.stamp(start + timedelta(days=32)), completed_steps=32,
              terminated=False, truncated=True)
    effective = {k: v for k, v in native_recipe.items() if k not in ("policy", "policy_kwargs", "train_freq")}
    effective.update(native_class=audit.ALGORITHMS[algorithm], observation_shape=[13], action_shape=[1])
    training = dict(symbol="BTCUSDT", algorithm=algorithm, seed=0, actual_steps=steps, effective_parameters=effective,
                    counters=dict(environment_steps=steps, gradient_update_rounds=updates,
                                  replay_add_calls=steps, replay_transitions=steps,
                                  optimizer_steps={"actor": updates // 2, "critic": updates} if algorithm == "td3" else
                                  {"actor": updates, "critic": updates, "entropy": updates}),
                    training_window_quality=dict(requested_train_days=1827, episode_days=32,
                                                 eligible_windows=1796, covered_training_days=1827),
                    training_episode_history=[deepcopy(ep) for _ in range(steps // 32)])
    model_dir = tmp_path / "model"
    checkpoint = model_dir / "checkpoint"
    checkpoint.mkdir(parents=True)
    native_data = {k: v for k, v in native_recipe.items() if k != "policy"}
    if algorithm != "td3":
        native_data["policy_kwargs"] = dict(native_data["policy_kwargs"], use_sde=False)
    with zipfile.ZipFile(checkpoint / "model.zip", "w") as zipped:
        zipped.writestr("data", json.dumps(native_data))
    manifest = dict(symbol="BTCUSDT", algorithm=algorithm, seed=0, recipe_id=context["recipe_id"],
                    rolling_context=context, recipe=native_recipe, training=training,
                    observation_shape=[13], action_shape=[1], independent_account=True,
                    multi_model_target_average=False, no_optimizer_or_replay_resume=True,
                    parameter_sha256="c" * 64, adapter_sha256="d" * 64,
                    checkpoint_files={"model.zip": audit.sha(checkpoint / "model.zip")})
    (checkpoint / "rolling_model.json").write_text(json.dumps(manifest), encoding="utf-8")
    record = dict(symbol="BTCUSDT", algorithm=algorithm, seed=0, recipe_id=context["recipe_id"],
                  context=context, checkpoint=manifest, training=training,
                  maximum_input_hour=audit.stamp(audit.utc(context["train_end"]) - timedelta(hours=1)),
                  input_market_sha256="e" * 64,
                  files={"checkpoint/" + p.name: audit.sha(p) for p in checkpoint.iterdir()})
    return record, manifest, context, native_recipe, steps, model_dir


def refresh_fixture_files(record, manifest, model_dir):
    file = model_dir / "checkpoint/rolling_model.json"
    file.write_text(json.dumps(manifest), encoding="utf-8")
    record["checkpoint"] = manifest
    record["training"] = manifest["training"]
    record["files"] = {"checkpoint/" + p.name: audit.sha(p) for p in file.parent.iterdir()}


def test_calendar_counts_shared_versions_and_last_day():
    weekly = audit.schedule_versions(cadence_days=7)
    four_week = audit.schedule_versions(cadence_days=28)
    fixed = audit.schedule_versions(cadence_days=None)
    assert len(weekly) == 53 and len(four_week) == 14 and len(fixed) == 1
    assert four_week == weekly[::4]
    assert fixed == weekly[:1]
    assert weekly[-1]["version_index"] == 52
    assert weekly[-1]["effective_day"] == "2026-10-22T00:00:00Z"
    assert weekly[0]["train_end"] == "2025-10-22T00:00:00Z"
    assert all(audit.utc(row["effective_day"]) - audit.utc(row["train_end"]) == timedelta(days=1) for row in weekly)
    assert all(audit.utc(row["train_end"]) - audit.utc(row["train_start"]) == timedelta(days=1827) for row in weekly)


@pytest.mark.parametrize("kwargs", [dict(cadence_days=True), dict(cadence_days=14), dict(train_days=365),
                                     dict(lag_days=0), dict(preheat_days=199),
                                     dict(start="2025-10-23T00:00:00"), dict(start="2025-10-23T01:00:00Z")])
def test_calendar_rejects_changed_protocol_or_ambiguous_time(kwargs):
    with pytest.raises((audit.AuditError, ValueError)):
        audit.schedule_versions(**kwargs)


@pytest.mark.parametrize("field,value", [("seed", True), ("seed", 5), ("symbol", "ADAUSDT"),
                                        ("algorithm", "ppo"), ("recipe_id", "../sac"),
                                        ("protocol_sha256", "x" * 64)])
def test_context_refuses_wrong_ownership_seed_or_unbound_recipe(field, value):
    kwargs = dict(symbol="BTCUSDT", algorithm="sac", recipe_id="sac_default", seed=0,
                  protocol_sha256="a" * 64)
    kwargs[field] = value
    with pytest.raises(audit.AuditError):
        audit.model_context(audit.schedule_versions()[0], **kwargs)


def test_swap_is_financially_inert():
    before = snapshot()
    assert audit.audit_swap(before, deepcopy(before)) is True


@pytest.mark.parametrize("field,value", [("cash", 10000.), ("positions", {}), ("cost_basis", {}),
                                        ("realized_pnl", 0.), ("high_water", 11000.),
                                        ("fees", 0.), ("steps", 0), ("fills_digest", "b" * 64)])
def test_swap_detects_hidden_account_or_drawdown_reset(field, value):
    before, after = snapshot(), snapshot()
    after[field] = value
    with pytest.raises(audit.AuditError, match="reset or changed"):
        audit.audit_swap(before, after)


def test_swap_refuses_incomplete_snapshot_even_if_equal():
    bad = snapshot()
    del bad["cost_basis"]
    with pytest.raises(audit.AuditError, match="Complete"):
        audit.audit_swap(bad, deepcopy(bad))


def test_market_digest_ignores_other_coin_and_includes_every_own_input():
    index = pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC")
    data = {"hourly_" + f: pd.DataFrame({"BTCUSDT": np.arange(8, dtype=float) + 1,
                                        "ETHUSDT": np.full(8, 100.)}, index=index)
            for f in ("open", "high", "low", "close", "volume")}
    digest = audit.input_market_digest(data, "BTCUSDT")
    other = {k: v.copy() for k, v in data.items()}
    other["hourly_close"]["ETHUSDT"] *= 10
    assert audit.input_market_digest(other, "BTCUSDT") == digest
    other["hourly_volume"].loc[index[0], "BTCUSDT"] += .01
    assert audit.input_market_digest(other, "BTCUSDT") != digest


def test_market_digest_rejects_ohlcv_misalignment():
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    data = {"hourly_" + f: pd.DataFrame({"BTCUSDT": [1., 2.]}, index=index)
            for f in ("open", "high", "low", "close", "volume")}
    data["hourly_volume"].index += pd.Timedelta(hours=1)
    with pytest.raises(audit.AuditError, match="align"):
        audit.input_market_digest(data, "BTCUSDT")


@pytest.mark.parametrize("algorithm", ["sac", "tqc", "td3"])
def test_native_complete_model_evidence_passes(tmp_path, algorithm):
    args = model_fixture(tmp_path, algorithm)
    assert audit.audit_model_evidence(*args, expected_market_sha256="e" * 64)["passed"]


@pytest.mark.parametrize("mutation,match", [
    ("future_hour", "unavailable"), ("wrong_crop", "market crop"),
    ("steps", "learner interactions"), ("optimizer", "optimizer rounds"),
    ("episode_future", "physical training bounds"), ("episode_terminal", "truncations"),
    ("recipe", "complete native recipe"), ("resume", "freshly trained"),
    ("native_gamma", "serialized parameter"), ("native_sde", "use_sde=False"),
])
def test_training_gate_rejects_specific_real_failure_paths(tmp_path, mutation, match):
    record, manifest, context, native_recipe, steps, model_dir = model_fixture(tmp_path)
    if mutation == "future_hour":
        record["maximum_input_hour"] = context["train_end"]
    elif mutation == "wrong_crop":
        record["input_market_sha256"] = "f" * 64
    elif mutation == "steps":
        manifest["training"]["actual_steps"] -= 1
    elif mutation == "optimizer":
        manifest["training"]["counters"]["optimizer_steps"]["actor"] -= 1
    elif mutation == "episode_future":
        manifest["training"]["training_episode_history"][0]["end_exclusive"] = audit.stamp(audit.utc(context["train_end"]) + timedelta(days=1))
    elif mutation == "episode_terminal":
        manifest["training"]["training_episode_history"][0]["terminated"] = True
    elif mutation == "recipe":
        manifest["recipe"] = dict(native_recipe, gamma=.99)
    elif mutation == "resume":
        manifest["no_optimizer_or_replay_resume"] = False
    else:
        data = {k: v for k, v in native_recipe.items() if k != "policy"}
        if mutation == "native_gamma":
            data["gamma"] = .99
        else:
            data["policy_kwargs"] = dict(data["policy_kwargs"], use_sde=True)
        path = model_dir / "checkpoint/model.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data", json.dumps(data))
        manifest["checkpoint_files"]["model.zip"] = audit.sha(path)
    refresh_fixture_files(record, manifest, model_dir)
    with pytest.raises(audit.AuditError, match=match):
        audit.audit_model_evidence(record, manifest, context, native_recipe, steps, model_dir,
                                   expected_market_sha256="e" * 64)


def test_checkpoint_changed_even_when_manifest_claims_old_identity(tmp_path):
    args = model_fixture(tmp_path)
    with (args[-1] / "checkpoint/model.zip").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(audit.AuditError, match="checkpoint hash"):
        audit.audit_model_evidence(*args)


def test_safe_paths_reject_escape(tmp_path):
    for name in ("../secret", "C:/secret", "/secret", "a//b", "a/./b"):
        with pytest.raises(audit.AuditError):
            audit.safe_path(tmp_path, name)


@pytest.fixture(scope="module")
def synthetic_account(tmp_path_factory):
    """Actual frozen env and serialization, with inert policy-loading markers."""
    from scripts.single_pair_rl_env import SinglePairEnv
    from scripts.crypto_rl_reporting import save_result, summarize, canonical_csv
    from scripts.backtest_rolling_single_pair import snapshot as real_snapshot
    root = tmp_path_factory.mktemp("real_rolling_account")
    start, end = map(pd.Timestamp, audit.TEST)
    index = pd.date_range(start - pd.Timedelta(days=210), end, freq="h", inclusive="left")
    level = 100. + np.arange(len(index)) * .001 + np.sin(np.arange(len(index)) / 71.)
    data = {"assets": {"BTCUSDT": {"symbol": "BTC"}}}
    for field in ("open", "high", "low", "close", "volume"):
        values = np.full(len(index), 100.) if field == "volume" else level * {"open": 1., "high": 1.01, "low": .99, "close": 1.001}[field]
        data["hourly_" + field] = pd.DataFrame({"BTCUSDT": values, "ETHUSDT": np.full(len(index), np.nan)}, index=index)
    env = SinglePairEnv(data, "BTCUSDT", *audit.TEST, costs=audit.COSTS["base"], capital=10000.,
                        rebalance_band=.025, drawdown_penalty=.25)
    obs, _ = env.reset(seed=0)
    contexts = {v["version_index"]: audit.model_context(v, symbol="BTCUSDT", algorithm="sac", recipe_id="sac_default", seed=0, protocol_sha256="b" * 64)
                for v in audit.schedule_versions()}
    pins = {("BTCUSDT", "sac_default", 0, version): {"manifest_sha256": hashlib.sha256(str(version).encode()).hexdigest()}
            for version in contexts}
    actions, swaps, last = [], [], None
    done = False
    while not done:
        day = env.current_date
        offset = (day - start).days
        version = offset // 7
        pin = pins[("BTCUSDT", "sac_default", 0, version)]
        if version != last:
            before = real_snapshot(env)
            swaps.append(dict(date=str(day), version_index=version, model_manifest_sha256=pin["manifest_sha256"],
                              context=contexts[version], before=before, after=deepcopy(before)))
            last = version
        observed = np.asarray(obs, dtype="<f4")
        # Real turns, costs, full exits and re-entry at several model boundaries.
        act = np.asarray([0.7 if (offset // 11) % 3 else -0.2], dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(act)
        actions.append(dict(date=str(day), action=act.tolist(), target=info["target_weights"].tolist(),
                            version_index=version, model_manifest_sha256=pin["manifest_sha256"],
                            observation=observed.tolist(), observation_sha256=hashlib.sha256(observed.tobytes()).hexdigest()))
        done = terminated or truncated
    result = env.get_result()
    canonical = root / "evaluation/test/BTCUSDT/sac/seed0/base"
    save_result(canonical, result)
    row, daily, monthly = summarize(result, run_id="test__base__BTCUSDT__sac__0", strategy="sac", seed=0,
                                    scenario="base", split="test", start=start, end=end, capital=10000., **audit.COSTS["base"])
    row.update(symbol="BTCUSDT", algorithm="sac", recipe_id="sac_default", cadence="every_week",
               account_id="BTCUSDT__sac_default__every_week__seed0__base", primary_seed=True,
               ensemble=False, independent_capital=True)
    for name, document in (("summary.json", row), ("actions.json", actions), ("swaps.json", swaps)):
        (canonical / name).write_text(json.dumps(document, allow_nan=False), encoding="utf-8")
    (canonical / "daily_equity.csv").write_bytes(canonical_csv(daily))
    (canonical / "monthly_returns.csv").write_bytes(canonical_csv(monthly))
    repeated = root / "repeat/test/BTCUSDT/sac/seed0/base"
    shutil.copytree(canonical, repeated)
    oracle = audit._load_oracle(Path(__file__).resolve().parents[1])
    market = oracle._market(data, "BTCUSDT", start, end)
    features = audit.own_market_features(data, "BTCUSDT", market["days"])
    return root, data, oracle, market, features, pins


def test_real_full_year_account_financial_oracle_and_continuous_swaps(synthetic_account):
    root, data, oracle, market, features, pins = synthetic_account
    finance = audit.audit_financial_case(root, market, "BTCUSDT", "sac", 0, "base", oracle=oracle)
    assert finance["passed"], finance["failures"][:3]
    result = audit.audit_account_schedule(root, "BTCUSDT", "sac_default", 0, "every_week", "base",
        spec={"test": list(audit.TEST), "recipes": {"sac_default": {"algorithm": "sac"}}},
        protocol_sha256="b" * 64, market=market, features=features, model_pins=pins, oracle=oracle)
    assert result["failure_count"] == 0, result["failures"][:3]


def test_future_changes_and_other_coin_cannot_change_earlier_features(synthetic_account):
    _, data, _, market, features, _ = synthetic_account
    other = {key: value.copy() if isinstance(value, pd.DataFrame) else deepcopy(value) for key, value in data.items()}
    boundary = pd.Timestamp("2026-06-01T00:00:00Z")
    for key, frame in other.items():
        if isinstance(frame, pd.DataFrame):
            frame["ETHUSDT"] = 1e12
            frame.loc[frame.index >= boundary, "BTCUSDT"] *= 10
    changed = audit.own_market_features(other, "BTCUSDT", market["days"])
    before = market["days"] <= boundary
    np.testing.assert_array_equal(changed[before], features[before])


@pytest.mark.parametrize("mutation,check_fragment", [("future_observation", "causal observation"),
                                                   ("wrong_policy", "shared scheduled model"),
                                                   ("hidden_reset", "financial state invariant"),
                                                   ("fake_high_water", "continuous historical high water")])
def test_real_saved_account_rejects_policy_future_input_and_hidden_reset(tmp_path, synthetic_account, mutation, check_fragment):
    root, _, oracle, market, features, pins = synthetic_account
    changed = tmp_path / "corrupt_account"
    shutil.copytree(root, changed)
    case = Path("evaluation/test/BTCUSDT/sac/seed0/base")
    if mutation in ("future_observation", "wrong_policy"):
        file = changed / case / "actions.json"
        rows = json.loads(file.read_text())
        if mutation == "future_observation":
            rows[30]["observation"][0] += .02
            rows[30]["observation_sha256"] = hashlib.sha256(np.asarray(rows[30]["observation"], dtype="<f4").tobytes()).hexdigest()
        else:
            rows[30]["version_index"] += 1
    else:
        file = changed / case / "swaps.json"
        rows = json.loads(file.read_text())
        if mutation == "hidden_reset":
            rows[3]["after"]["cash"] = 10000.
        else:
            rows[3]["before"]["high_water"] += 10.
            rows[3]["after"]["high_water"] += 10.
    file.write_text(json.dumps(rows), encoding="utf-8")
    result = audit.audit_account_schedule(changed, "BTCUSDT", "sac_default", 0, "every_week", "base",
        spec={"test": list(audit.TEST), "recipes": {"sac_default": {"algorithm": "sac"}}},
        protocol_sha256="b" * 64, market=market, features=features, model_pins=pins, oracle=oracle)
    assert result["failure_count"] > 0
    assert any(check_fragment in item["check"] for item in result["failures"]), result["failures"][:5]


def protocol_fixture():
    return dict(test=list(audit.TEST), seeds=list(range(5)), cadences=deepcopy(audit.CADENCES), costs=deepcopy(audit.COSTS),
                capital_per_account=10000., capital_shared_between_pairs=False, ensemble=False,
                selection_uses_test=False, fresh_holdout=False, test_year_already_observed=True,
                train_days=1827, lag_days=1, preheat_days=200,
                selected={"BTCUSDT": ["td3_default", "sac_default", "sac_gamma099"],
                          "ETHUSDT": ["sac_default", "tqc_default", "sac_gamma099"]},
                recipes={name: dict(algorithm=algorithm, parameters=recipe(algorithm), timesteps=65536 if algorithm == "td3" else 32768)
                         for name, algorithm in (("sac_default", "sac"), ("sac_gamma099", "sac"),
                                                 ("td3_default", "td3"), ("tqc_default", "tqc"))},
                expected_models=1590, expected_accounts=180,
                dataset_sha256="a" * 64, dataset_manifest_sha256="b" * 64,
                inventory_sha256="c" * 64, selection_sha256="d" * 64,
                sources={audit.SOURCE_ORACLE: audit.ORACLE_SHA256})


def test_full_grid_math_and_contract_are_independent_of_runner():
    spec = protocol_fixture()
    assert audit.validate_protocol(spec)
    contexts = audit._all_expected_models(spec, "e" * 64)
    assert len(contexts) == 1590
    assert len({tuple(sorted(c.items())) for c in contexts}) == 1590
    assert len([c for c in contexts if c["version_index"] == 0]) == 30


@pytest.mark.parametrize("field,value", [("selection_uses_test", True), ("capital_shared_between_pairs", True),
                                        ("fresh_holdout", True), ("seeds", [False, 1, 2, 3, 4]),
                                        ("expected_models", 1589), ("expected_accounts", 179),
                                        ("lag_days", 0), ("dataset_manifest_sha256", "unbound")])
def test_protocol_rejects_incomplete_grid_leakage_and_misleading_claims(field, value):
    spec = protocol_fixture()
    spec[field] = value
    with pytest.raises(audit.AuditError):
        audit.validate_protocol(spec)


def test_failed_real_study_attempt_never_writes_success_gate_or_changes_prior_failure(tmp_path):
    study = tmp_path / "incomplete"
    study.mkdir()
    (study / "protocol.json").write_text(json.dumps({"test": list(audit.TEST)}))
    old_failure = study / "audit_failed_old.json"
    old_failure.write_text('{"passed":false,"reason":"preserve"}')
    before = old_failure.read_bytes()
    result = audit.audit_study(study, tmp_path / "missing_market.pkl.gz", max_workers=1)
    assert result["passed"] is False and result["failure_count"] > 0
    assert result["models_checked"] == result["accounts_checked"] == result["repeats_checked"] == 0
    assert not (study / "completion_gate.json").exists()
    assert old_failure.read_bytes() == before
    assert len(list(study.glob("audit_failed_*.json"))) == 2
