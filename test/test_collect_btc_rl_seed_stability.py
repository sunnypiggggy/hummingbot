"""BTC/30-seed export population checks with synthetic, nonfinancial bytes.

The actual generated remote-export program executes in a local temporary tree.
No SSH, model, environment, trainer, or source study is accessed. The unchanged
base collector's separate safety suite owns general transfer/path tests.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts import collect_btc_rl_seed_stability as collector


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def completed_tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_nonfinancial_btc30") / "study"
    root.mkdir()
    spec = {"symbols": ["BTCUSDT"], "algorithms": ["sac", "td3"], "seeds": list(range(30)),
            "training_model_count": 60, "costs": {"base": {"fee": .001, "slippage": .0002},
            "stress": {"fee": .002, "slippage": .0005}}, "baselines": ["cash", "buy_hold", "sma200_trend", "momentum180"],
            "ensemble": False, "capital_per_account": 10000, "capital_shared_between_pairs": False,
            "synthetic_nonfinancial_fixture_only": True}
    dump(root / "protocol.json", spec)
    dump(root / "pipeline_status.json", {"status": "completed", "synthetic_nonfinancial_fixture_only": True})
    dump(root / "pipeline_launch.json", {"pid": 987654321, "synthetic_nonfinancial_fixture_only": True})
    counter = {"passed": True, "failure_count": 0, "failures": [], "models_checked": 60,
               "core_protocol_sha256": collector.transfer.sha(root / "protocol.json"), "models": []}
    for algorithm in spec["algorithms"]:
        for seed in spec["seeds"]:
            name = f"training/BTCUSDT/{algorithm}/seed{seed}/record.json"
            record = {"symbol": "BTCUSDT", "algorithm": algorithm, "seed": seed,
                      "training": {"counters": {"optimizer_steps": {"actor": 32512 if algorithm == "td3" else 32256}}},
                      "synthetic_nonfinancial_fixture_only": True}
            dump(root / name, record)
            counter["models"].append({"symbol": "BTCUSDT", "algorithm": algorithm, "seed": seed,
                                      "record_sha256": collector.transfer.sha(root / name)})
    dump(root / "training_counter_audit.json", counter)
    files = ["equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json",
             "fingerprints.json", "actions.json", "summary.json", "daily_equity.csv", "monthly_returns.csv"]
    identities = [(a, seed, cost) for a in spec["algorithms"] for seed in spec["seeds"] for cost in spec["costs"]]
    identities += [(a, None, cost) for a in spec["baselines"] for cost in spec["costs"]]
    assert len(identities) == 128
    for split in ("validation", "test"):
        (root / (split + "_summary.csv")).write_bytes(b"SYNTHETIC NONFINANCIAL SUMMARY ONLY\n")
        dump(root / (split + "_complete.json"), {"synthetic_nonfinancial_fixture_only": True})
        inputs = {n: collector.transfer.sha(root / n) for n in ("protocol.json", split + "_summary.csv", split + "_complete.json")}
        cases = []
        for algorithm, seed, cost in identities:
            directory = f"evaluation/{split}/BTCUSDT/{algorithm}/{'baseline' if seed is None else 'seed'+str(seed)}/{cost}"
            hashes = {}
            for prefix in (directory, directory.replace("evaluation/", "repeat/", 1)):
                for name in files:
                    path = root / prefix / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(("SYNTHETIC-NONFINANCIAL:" + prefix + "/" + name).encode())
                    hashes[path.relative_to(root).as_posix()] = collector.transfer.sha(path)
            cases.append({"relative_path": directory, "passed": True, "failure_count": 0, "failures": [],
                          "checks_total": 1, "inputs_sha256": hashes})
        dump(root / (split + "_audit.json"), {"phase": split, "passed": True, "failure_count": 0, "failures": [],
            "checks_total": 128, "expected_accounts": 128, "accounts_checked": 128, "accounts_count": 128,
            "repeats_checked": 128, "cases": cases, "protocol_sha256": collector.transfer.sha(root / "protocol.json"),
            "summary_sha256": collector.transfer.sha(root / (split + "_summary.csv")), "inputs_sha256": inputs})
        dump(root / (split + "_seed_contract_audit.json"), {"passed": True, "failure_count": 0, "failures": [],
            "checks_total": 1, "core_financial_audit_passed": True,
            "core_financial_audit_sha256": collector.transfer.sha(root / (split + "_audit.json")),
            "inputs_sha256": {"protocol.json": collector.transfer.sha(root / "protocol.json"),
                              "training_counter_audit.json": collector.transfer.sha(root / "training_counter_audit.json")}})
    return root


@pytest.fixture
def study(completed_tree):
    names = ["validation_audit.json", "validation_seed_contract_audit.json", "test_seed_contract_audit.json",
             "training/BTCUSDT/td3/seed29/record.json"]
    original = {name: (completed_tree / name).read_bytes() for name in names}
    yield completed_tree
    for name, body in original.items():
        (completed_tree / name).write_bytes(body)


def export_locally(root):
    program = collector.extended_export(collector.transfer.REMOTE_EXPORT).replace("REMOTE_ROOT", repr(str(root)))
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(program, "<synthetic-nonfinancial-btc30-export>", "exec"), {})
    return json.loads(output.getvalue())


def test_complete_btc30_export_preserves_all_bound_bytes(study):
    assert collector.transfer.sha(collector.transfer.__file__) == collector.BASE_COLLECTOR_SHA256
    receipt = export_locally(study)
    assert receipt["archive_sha256"] == collector.transfer.sha(receipt["archive"])
    assert receipt["protocol_sha256"] == collector.transfer.sha(study / "protocol.json")
    assert all(collector.transfer.sha(study / name) == expected for name, expected in receipt["files"].items())
    assert len(receipt["files"]) > 5120 + 60
    with tarfile.open(receipt["archive"], "r:gz") as stream:
        members = stream.getmembers()
        assert {m.name for m in members} == set(receipt["files"])
        assert all(m.isfile() for m in members)


@pytest.mark.parametrize("damage", ["missing_seed_case", "wrong_contract_binding", "false_contract", "training_actor_record_hash"])
def test_incomplete_or_unbound_eth_extension_never_exports(study, damage):
    if damage == "missing_seed_case":
        path = study / "validation_audit.json"
        value = json.loads(path.read_text())
        value["cases"] = [case for case in value["cases"] if case["relative_path"] != "evaluation/validation/BTCUSDT/td3/seed29/base"]
    elif damage == "wrong_contract_binding":
        path = study / "test_seed_contract_audit.json"
        value = json.loads(path.read_text())
        value["core_financial_audit_sha256"] = "0" * 64
    elif damage == "false_contract":
        path = study / "validation_seed_contract_audit.json"
        value = json.loads(path.read_text())
        value["passed"] = False
    else:
        path = study / "training/BTCUSDT/td3/seed29/record.json"
        value = json.loads(path.read_text())
        value["training"]["counters"]["optimizer_steps"]["actor"] += 1
    dump(path, value)
    before = set(study.parent.glob("*_audited_*.tar.gz"))
    with pytest.raises(AssertionError):
        export_locally(study)
    assert set(study.parent.glob("*_audited_*.tar.gz")) == before
