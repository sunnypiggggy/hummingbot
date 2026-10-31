"""A 1,024-step native save/load check, explicitly ineligible for formal results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    output, core_root = Path(args.output).resolve(), Path(args.core_root).resolve()
    adapter_path = Path(__file__).with_name("sac_validation_optimization_models.py")
    if sha(adapter_path) != args.expected_adapter_sha256:
        raise ValueError("Freeze the tested adapter before running the preflight")
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    if sha(plan_path) != "8ab80ceb21720691af055c7ed3505b69997087437f87315484ba2c64af625c36":
        raise ValueError("The predeclared search plan changed")
    if output.exists():
        raise FileExistsError("Preserve every previous preflight attempt")
    for name, expected in plan["frozen_core_sources"].items():
        if sha(core_root / name) != expected:
            raise ValueError("The original financial or training helper changed")
    if sha(args.dataset) != plan["dataset_sha256"]:
        raise ValueError("The preflight must use the original training data")
    output.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    # Both roots are explicit. Financial helpers resolve only to the old source.
    for root in (Path(__file__).resolve().parents[1], core_root):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    from scripts.backtest_btceth_rl import load_dataset
    from scripts.backtest_single_pair_rl import cropped_data
    from scripts.single_pair_rl_env import SinglePairEnv
    models = importlib.import_module("scripts.sac_validation_optimization_models")
    if Path(models.__file__).resolve() != adapter_path or sha(models.__file__) != args.expected_adapter_sha256:
        raise ValueError("Preflight imported another adapter")
    import numpy as np
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one isolated real CUDA GPU is required")
    started = datetime.now(timezone.utc).isoformat()
    data = cropped_data(load_dataset(args.dataset), plan["train"][1])
    env = SinglePairEnv(data, args.symbol, *plan["train"], training=True, episode_days=32,
        record_history=False, capital=10000., drawdown_penalty=.25, rebalance_band=.025)
    model = models.make_model(env, seed=29, candidate_id=args.candidate, stage="preflight",
        parent_protocol_sha256=sha(plan_path), core_root=core_root, device="cuda:0")
    training = models.train_model(model, env, core_root=core_root, progress_path=output / "progress.jsonl")
    probes = np.random.default_rng(2047).normal(0., .1, size=(16, 13)).astype("float32")
    probes[:, 10:] = [0., 1., 0.]
    before = np.asarray([models.predict(model, x, core_root=core_root) for x in probes], dtype="<f4")
    manifest = models.save_model(model, output / "checkpoint", core_root=core_root)
    restored = models.load_model(output / "checkpoint", symbol=args.symbol, seed=29, candidate_id=args.candidate,
        stage="preflight", parent_protocol_sha256=sha(plan_path), core_root=core_root, device="cuda:0")
    after = np.asarray([models.predict(restored, x, core_root=core_root) for x in probes], dtype="<f4")
    if before.tobytes() != after.tobytes() or manifest["nonfinancial_preflight"] is not True:
        raise ValueError("Real native predictions changed after checkpoint restoration")
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True).splitlines()
    own_rows = [line for line in gpu_rows if line.split(",")[0].strip() == str(os.getpid())]
    if not own_rows or any(line.split(",")[1].strip() != args.gpu_uuid for line in own_rows):
        raise ValueError("The actual process did not execute on its declared physical GPU")
    if sha(adapter_path) != args.expected_adapter_sha256:
        raise ValueError("The adapter changed during preflight")
    result = {"passed": True, "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
        "nonfinancial_preflight": True, "formal_result_eligible": False, "symbol": args.symbol,
        "candidate_id": args.candidate, "seed": 29, "pid": os.getpid(), "actual_gpu_uuid": args.gpu_uuid,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "visible_cuda_devices": torch.cuda.device_count(),
        "physical_process_rows": own_rows, "adapter_sha256": args.expected_adapter_sha256,
        "smoke_source_sha256": sha(__file__), "search_plan_sha256": sha(plan_path), "dataset_sha256": sha(args.dataset),
        "maximum_input_hour": str(data["hourly_close"].index[-1]), "training": training,
        "checkpoint_manifest_sha256": sha(output / "checkpoint/single_pair_model.json"),
        "actual_recipe": manifest["recipe"], "probe_count": len(probes),
        "probe_observation_sha256": hashlib.sha256(probes.astype("<f4").tobytes()).hexdigest(),
        "prediction_sha256": hashlib.sha256(before.tobytes()).hexdigest(), "save_load_predictions_byte_equal": True,
        "no_production_change": True}
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {key: result[key] for key in ("passed", "nonfinancial_preflight", "symbol", "candidate_id", "actual_gpu_uuid", "save_load_predictions_byte_equal")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"), required=True)
    parser.add_argument("--candidate", choices=("c00_baseline", "c03_lr1e4_gamma099"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args)))


if __name__ == "__main__":
    main()
