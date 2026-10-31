"""Small native CPU proofs; these fixtures are not research backtest outcomes."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import sac_validation_optimization_models as models

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA = "8ab80ceb21720691af055c7ed3505b69997087437f87315484ba2c64af625c36"


def synthetic_env(symbol):
    import gymnasium as gym

    class NonfinancialShapeFixture(gym.Env):
        observation_space = gym.spaces.Box(-10., 10., (13,), dtype=np.float32)
        action_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)
        SYMBOLS = (symbol,)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.position = 0
            return np.zeros(13, dtype=np.float32), {}

        def step(self, action):
            self.position += 1
            values = np.array([self.position / 32, float(action[0]), *[np.sin(self.position + i) * .1 for i in range(11)]], dtype=np.float32)
            return values, float(action[0]) * .001, False, self.position == 32, {}
    return NonfinancialShapeFixture()


def test_only_learning_rate_gamma_change():
    original = models.base_models(ROOT).recipe("sac")
    for candidate in models.CANDIDATE_OVERRIDES:
        value = models.recipe(candidate, ROOT)
        assert {k: v for k, v in value.items() if k not in ("learning_rate", "gamma")} == {
            k: v for k, v in original.items() if k not in ("learning_rate", "gamma")}
    assert models.recipe("c00_baseline", ROOT) == original
    with pytest.raises(ValueError):
        models.recipe("c04_unplanned", ROOT)


@pytest.mark.parametrize("stage,seed,accepted", [("tuning", 100, True), ("tuning", 104, True),
    ("tuning", 0, False), ("final", 29, True), ("final", 100, False), ("preflight", 29, True),
    ("preflight", 100, False), ("anything", 29, False)])
def test_disjoint_seed_scopes(stage, seed, accepted):
    if accepted:
        assert models.context(stage, "BTCUSDT", seed, "c00_baseline", PLAN_SHA)["seed"] == seed
    else:
        with pytest.raises(ValueError):
            models.context(stage, "BTCUSDT", seed, "c00_baseline", PLAN_SHA)


@pytest.mark.parametrize("candidate,symbol", [("c00_baseline", "BTCUSDT"), ("c03_lr1e4_gamma099", "ETHUSDT")])
def test_native_preflight_truthful_recipe_counters_and_restoration(tmp_path, monkeypatch, candidate, symbol):
    base = models.base_models(ROOT)
    monkeypatch.setattr(base, "SEEDS", base.SEEDS)
    env = synthetic_env(symbol)
    model = models.make_model(env, seed=29, candidate_id=candidate, stage="preflight",
        parent_protocol_sha256=PLAN_SHA, core_root=ROOT, device="cpu")
    training = models.train_model(model, env, core_root=ROOT, progress_path=tmp_path / "progress.jsonl")
    assert training["actual_steps"] == 1024
    assert training["counters"]["optimizer_steps"] == {"actor": 512, "critic": 512, "entropy": 512}
    saved = models.save_model(model, tmp_path / "checkpoint", core_root=ROOT)
    assert saved["recipe"] == models.recipe(candidate, ROOT)
    assert saved["adapter_sha256"] == models.sha(models.__file__) and saved["base_adapter_sha256"] == models.BASE_ADAPTER_SHA256
    assert saved["nonfinancial_preflight"] is True and saved["model_context"]["stage"] == "preflight"
    loaded = models.load_model(tmp_path / "checkpoint", symbol=symbol, seed=29, candidate_id=candidate,
        stage="preflight", parent_protocol_sha256=PLAN_SHA, core_root=ROOT, device="cpu")
    assert models.parameter_digest(model, ROOT) == models.parameter_digest(loaded, ROOT)
    for i in range(16):
        obs = np.array([np.sin(i + j) * .1 for j in range(13)], dtype=np.float32)
        np.testing.assert_array_equal(models.predict(model, obs, core_root=ROOT), models.predict(loaded, obs, core_root=ROOT))
    with pytest.raises(FileExistsError):
        models.save_model(model, tmp_path / "checkpoint", core_root=ROOT)
    if candidate == "c00_baseline":
        original_env = synthetic_env(symbol)
        base.SEEDS = (29,)
        old = base.make_model("sac", original_env, seed=29, device="cpu")
        original_training = base.train_model(old, original_env, steps=1024, seed=29)
        assert base.parameter_digest(old, "sac") == models.parameter_digest(model, ROOT)
        assert original_training["transition_sha256"] == training["transition_sha256"]
    path = tmp_path / "checkpoint/single_pair_model.json"
    edited = json.loads(path.read_text())
    edited["recipe"]["gamma"] = .98
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ValueError):
        models.load_model(path.parent, symbol=symbol, seed=29, candidate_id=candidate, stage="preflight",
            parent_protocol_sha256=PLAN_SHA, core_root=ROOT, device="cpu")


@pytest.mark.parametrize("key", ["learning_rate", "gamma", "tau", "learning_starts", "batch_size", "buffer_size", "gradient_steps", "ent_coef"])
def test_actual_native_parameters_cannot_be_relabelled(key):
    native = models.recipe("c03_lr1e4_gamma099", ROOT)
    parameters = {k: native[k] for k in ("learning_rate", "gamma", "tau", "learning_starts", "batch_size", "buffer_size", "gradient_steps", "ent_coef")}
    parameters.update(observation_shape=[13], action_shape=[1], native_class="SAC", train_freq=[1, "step"])
    training = {"actual_steps": 32768, "effective_parameters": parameters, "counters": {
        "environment_steps": 32768, "gradient_update_rounds": 32256, "optimizer_steps": {"actor": 32256, "critic": 32256, "entropy": 32256},
        "replay_add_calls": 32768, "replay_transitions": 32768}}
    models.check_effective_parameters(training, native)
    corrupted = deepcopy(training)
    corrupted["effective_parameters"][key] = "wrong"
    with pytest.raises(ValueError):
        models.check_effective_parameters(corrupted, native)
