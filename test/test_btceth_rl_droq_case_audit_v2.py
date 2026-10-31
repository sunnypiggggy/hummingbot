"""Real serialization contract for bf33's (run, DataFrame, monthly-dict)."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd

from scripts import btceth_rl_droq_case_audit_v2 as worker
from scripts import audit_btceth_rl_droq as bf
from scripts.crypto_rl_reporting import canonical_csv


def test_original_monthly_dict_is_serialized_as_json_without_dataframe_mock(tmp_path, monkeypatch):
    case = tmp_path / "original_case"; case.mkdir()
    for name in bf.CASE_FILES:
        (case / name).write_bytes(b"Explicit synthetic case bytes, not financial evidence")
    monthly = {"2025-01": np.float64(.0123456789012345), "2025-02": -.02}
    history = pd.DataFrame({"timestamp": ["2025-01-01T00:00:00Z"], "equity": [10000.]})
    run = {"run_id": "synthetic", "inventory_replay": {"verified": True}, "hours": 24}
    fake = SimpleNamespace(CASE_FILES=bf.CASE_FILES, Checks=bf.Checks,
        audit_case=lambda *args, **kwargs: (run, history, monthly))
    monkeypatch.setattr(worker, "_STATE", dict(bf=fake, dataset={}, features={}, records={},
        source_sha256={"synthetic": "not_a_source_pin"}, dataset_sha256="synthetic_only",
        original_gpus=[], canonical_csv=canonical_csv))
    destination = tmp_path / "new_receipt"
    value = worker.audit_case_worker(dict(directory=str(case), row={"run_id": "synthetic"},
        case_sha256={name: worker.sha(case / name) for name in bf.CASE_FILES},
        split="validation", family="raw", receipt_directory=str(destination), directory_relative="synthetic"))
    assert value["passed"] is True
    assert set(path.name for path in destination.iterdir()) == {"run.json", "history.csv", "monthly.json", "receipt.json"}
    assert json.loads((destination / "monthly.json").read_text()) == {k: float(v) for k, v in monthly.items()}
    assert (destination / "history.csv").read_bytes() == canonical_csv(history)
    receipt = json.loads((destination / "receipt.json").read_text())
    assert set(receipt["result_sha256"]) == {"run.json", "history.csv", "monthly.json"}
    assert all(worker.sha(destination / name) == digest for name, digest in receipt["result_sha256"].items())
    assert receipt["wrapper_source_sha256"] == worker.sha(Path(worker.__file__))
