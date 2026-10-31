"""Freeze the bounded native implementation comparison before its training."""
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.backtest_rl_elegantrl import DEFAULT_OUTPUT, DATASETS, read, sha
from scripts.backtest_crypto_rl_top10 import COSTS, SEEDS, TRAIN_START, VAL_START, TEST_START, TEST_END
from scripts.crypto_rl_reporting import write_json


def main():
    target = DEFAULT_OUTPUT / "frozen_spec.json"
    if target.exists():
        raise FileExistsError("A frozen protocol must not be overwritten")
    manifest_path = DEFAULT_OUTPUT / "sources/training_source_manifest.json"
    manifest = read(manifest_path)
    vendor = Path(manifest["training_import_directory"]).parent
    common = dict(net_dims=[32, 32], gamma=.99, learning_rate=.0003, batch_size=64,
                  reward_scale=1., soft_update_tau=.005)
    ppo = {**common, "repeat_times": 10., "ratio_clip": .2, "lambda_gae_adv": .95,
           "lambda_entropy": 0.}
    sac = {**common, "repeat_times": 0., "num_ensembles": 2}
    td3 = {**common, "repeat_times": 0., "num_ensembles": 2, "update_freq": 2,
           "policy_noise_std": .1, "explore_noise_std": .05}
    spec = {"frozen_at": datetime.now(timezone.utc).isoformat(),
            "implementation": "ElegantRL official helloworld; unmodified algorithm updates",
            "upstream_repository": manifest["repository"], "upstream_commit": manifest["commit"],
            "vendor_path": str(vendor), "vendor_manifest": str(manifest_path),
            "vendor_file_sha256": {row["path"]: row["sha256"] for row in manifest["files"]},
            "dataset_sha256_by_variant": {name: sha(path) for name, path in DATASETS.items()},
            "train": [TRAIN_START, VAL_START], "validation": [VAL_START, TEST_START],
            "test": [TEST_START, TEST_END], "costs": COSTS, "capital": 10000,
            "cash_target_floor": .05, "single_coin_target_cap": .25,
            "train_episode_days": 128, "seeds": list(SEEDS), "timesteps_per_seed": 32768,
            "observation": "Unchanged 92 fixed-scale causal features; no fitted normalization",
            "action": "11 continuous scores, native tanh, unchanged cash-floor and 25% cap mapper",
            "reward": "100*log(after-cost next_NAV/current_NAV); base-cost train only; no extra reward scaling",
            "model_selection": "Final checkpoints of every seed; no selection by validation or test PNL",
            "ensemble": "Mean of five mapped target weights in one independent 10000USDT ledger",
            "parameter_search": False, "test_retraining": False,
            "prior_test_year_already_observed": True,
            "ex_ada_is_post_hoc_sensitivity": True,
            "budget_caveat": "Equal environment interactions do not equate gradient or compute budgets; PPO5120, SAC8000, TD3critic8000/actor4000 updates per seed",
            "algorithms": {
                "ppo": {"parameters": ppo, "horizon": 256,
                        "update_schedule": "40 minibatch updates per 256 transitions, total5120",
                        "native_details": "ReLU MLP, SmoothL1 critic, Gaussian raw actions then tanh; native clipped min surrogate; no extra gradient clipping; differs from SB3 defaults"},
                "sac": {"parameters": sac, "horizon": 256, "warmup_transitions_included_in_budget": 1024,
                        "update_schedule": "64 updates after each256 transitions from replay_size1024; repeat_times=(4096+.001)/cur_size; total8000; uniform replay capacity32768",
                        "native_details": "Two configured critic heads; native target_entropy=-log(11), alpha initialization exp(-1), actor objective uses target critics; sampled-noise tanh-corrected log probability"},
                "td3": {"parameters": td3, "horizon": 256, "warmup_transitions_included_in_budget": 1024,
                        "update_schedule": "64 updates after each256 transitions from replay_size1024; repeat_times=(4096+.001)/cur_size; total8000 critic/4000 actor; replay capacity32768",
                        "native_details": "Two configured critic heads, delayed actor every2 updates; target actions use the online actor with noise as implemented upstream, not a textbook-only reproduction"}},
            "adapter_changes": [
                "Use the unchanged audited spot ledger, not a bundled stock environment",
                "Custom collector saves actual raw PPO log probabilities; native stock collector omits their assignment",
                "At PPO timeouts add gamma*V(real terminal observation), then stop GAE recurrence; true terminals receive no bootstrap",
                "Off-policy replay stores explicit real next observations before any random reset; true-terminal-only bootstrap mask, all valid transitions retain critic loss",
                "Bound native off-policy update count independently of growing replay size; random warmup counted in total budget",
                "Disable gradients for value/advantage preparation and deterministic evaluation; upstream update methods enable them for optimizer steps",
                "No algorithm source edits, no new validation/test data in training, no production changes"],
            "reproduction": [sys.executable, "scripts/backtest_rl_elegantrl.py", "--workers", "4", "--phase", "all"],
            "production_changes_authorized": False}
    write_json(target, spec)
    print(str(target))


if __name__ == "__main__":
    main()
