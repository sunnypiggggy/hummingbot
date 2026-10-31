"""Nonfinancial export-contract fixtures; the economic oracle has separate tests."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from scripts import collect_sac_validation_optimization as collector


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def export_case(tmp_path, monkeypatch):
    root = tmp_path / "research" / "optimization"
    root.mkdir(parents=True)
    external = root.parent / "retained" / "original.json"
    write(external, {"nonfinancial_fixture": True})
    write(root / "pipeline_status.json", {"status": "completed"})
    write(root / "pipeline_launch.json", {"pid": 999999999})
    write(root / "protocol.json", {"symbols": ["BTCUSDT", "ETHUSDT"], "capital_per_account": 10000,
        "ensemble": False, "capital_shared_between_pairs": False,
        "costs": {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}}})
    write(root / "search_plan.json", {"nonfinancial_fixture_plan": True})
    monkeypatch.setattr(collector, "PLAN_SHA256", digest(root / "search_plan.json"))
    original_audit = root / "extension_source/scripts/audit_sac_validation_optimization.py"
    original_audit.parent.mkdir(parents=True)
    shutil.copyfile(collector.ROOT / "scripts/audit_sac_validation_optimization.py", original_audit)
    assert digest(original_audit) == collector.ORIGINAL_AUDITOR_SHA256
    write(root / "extension_manifest.json", {"nonfinancial_fixture_original_extension": True})
    parent_source = root / "audit_revision_source/scripts/audit_sac_validation_optimization_v2.py"
    parent_source.parent.mkdir(parents=True)
    shutil.copyfile(collector.ROOT / "scripts/audit_sac_validation_optimization_v2.py", parent_source)
    assert digest(parent_source) == collector.PARENT_AUDITOR_SHA256
    controller = root / "audit_revision_source/scripts/continue_sac_validation_optimization.py"
    controller.write_text("# nonfinancial fixture continuation controller\n")
    revision = {"schema_version": 1, "revision_id": "native_sac_policy_kwargs_default_false_v2",
        "reason_code": "sb3_sac_saved_policy_kwargs_use_sde_false", "original_protocol_sha256": digest(root / "protocol.json"),
        "search_plan_sha256": digest(root / "search_plan.json"), "original_audit_source_sha256": collector.ORIGINAL_AUDITOR_SHA256,
        "original_extension_manifest_sha256": digest(root / "extension_manifest.json"),
        "original_pipeline_launch_sha256": digest(root / "pipeline_launch.json"),
        "revised_audit_source_path": parent_source.relative_to(root).as_posix(), "revised_audit_source_sha256": digest(parent_source),
        "controller_source_path": controller.relative_to(root).as_posix(), "controller_source_sha256": digest(controller),
        "financial_math_changed": False, "model_recipe_changed": False, "training_restarted": False}
    write(root / "audit_revision.json", revision)
    write(root / "pipeline_launch_revision.json", {"pid": 999999998, "training_restarted": False,
        "original_pipeline_launch_sha256": digest(root / "pipeline_launch.json"),
        "audit_revision_sha256": digest(root / "audit_revision.json"),
        "controller_source_sha256": digest(controller), "revised_audit_source_sha256": digest(parent_source)})
    source = root / "audit_import_revision_source/scripts/audit_sac_validation_optimization_v3.py"
    source.parent.mkdir(parents=True)
    source.write_text("import json\ndef verify_completed_study(root):\n return json.loads((root/'completion_contract_audit.json').read_text())\n", encoding="utf-8")
    finisher = root / "audit_import_revision_source/scripts/finish_sac_fallback_audit_v3.py"
    finisher.write_text("# nonfinancial fixture fallback audit finisher\n")
    write(root / "selection.json", {"nonfinancial_fixture_frozen_selection": True})
    write(root / "selection_audit.json", {"nonfinancial_fixture_passed_selection": True})
    write(root / "final/validation_optimization_audit.json", {"passed": False, "failure_count": 1})
    import_revision = {"schema_version": 1, "revision_id": "retained_import_population_shadowing_v3",
        "reason_code": "import_manifest_expected_digest_shadowing", "original_protocol_sha256": digest(root / "protocol.json"),
        "search_plan_sha256": digest(root / "search_plan.json"), "parent_audit_revision_sha256": digest(root / "audit_revision.json"),
        "parent_auditor_source_sha256": digest(parent_source), "selection_sha256": digest(root / "selection.json"),
        "selection_audit_sha256": digest(root / "selection_audit.json"),
        "failed_final_validation_audit_sha256": digest(root / "final/validation_optimization_audit.json"),
        "revised_audit_source_path": source.relative_to(root).as_posix(), "revised_audit_source_sha256": digest(source),
        "finisher_source_path": finisher.relative_to(root).as_posix(), "finisher_source_sha256": digest(finisher),
        "financial_math_changed": False, "model_recipe_changed": False, "selection_changed": False, "training_restarted": False}
    write(root / "audit_import_revision.json", import_revision)
    write(root / "audit_import_revision_receipt.json", {"sha256": digest(root / "audit_import_revision.json")})
    write(root / "pipeline_launch_import_revision.json", {"pid": 999999997, "training_restarted": False,
        "parent_pipeline_launch_revision_sha256": digest(root / "pipeline_launch_revision.json"),
        "audit_import_revision_sha256": digest(root / "audit_import_revision.json"),
        "finisher_source_sha256": digest(finisher), "revised_audit_source_sha256": digest(source)})
    write(root / "final/test_summary.csv", {"nonfinancial_fixture_result": True})
    complete = {"passed": True, "checks_total": 3, "failure_count": 0, "failures": [],
        "protocol_sha256": digest(root / "protocol.json"), "expected_tuning_models": 40, "final_models": 60,
        "inputs_sha256": {n: digest(root / n) for n in ("protocol.json", "final/test_summary.csv")},
        "external_inputs_sha256": {str(external): digest(external)}}
    write(root / "completion_contract_audit.json", complete)
    template = collector.optimization_export(collector.transfer.REMOTE_EXPORT, digest(source))
    return root, external, complete, template.replace("REMOTE_ROOT", repr(str(root)))


def execute(source):
    exec(compile(source, "<nonfinancial-export-fixture>", "exec"), {})


def test_completed_export_is_exact_regular_snapshot(export_case, capsys):
    root, _, _, source = export_case
    execute(source)
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["archive_sha256"] == digest(Path(receipt["archive"]))
    assert receipt["completion_contract_audit_sha256"] == digest(root / "completion_contract_audit.json")
    assert receipt["external_inputs_sha256"]
    assert receipt["external_verified_sha256"] == receipt["external_inputs_sha256"]
    assert receipt["external_rechecked_after_archive"] is True
    assert receipt["verifier_source_sha256"] == digest(root / "audit_import_revision_source/scripts/audit_sac_validation_optimization_v3.py")
    assert receipt["audit_revision_sha256"] == digest(root / "audit_revision.json")
    assert receipt["audit_import_revision_sha256"] == digest(root / "audit_import_revision.json")
    with tarfile.open(receipt["archive"]) as archive:
        members = archive.getmembers()
        assert {m.name for m in members} == set(receipt["files"])
        assert all(m.isfile() for m in members)
        for member in members:
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == receipt["files"][member.name]


@pytest.mark.parametrize("change", ["running", "failed_gate", "missing_models", "changed_result", "changed_reference", "changed_plan", "changed_revision", "changed_controller", "changed_finisher", "changed_selection"])
def test_refuses_incomplete_or_changed_evidence(export_case, change):
    root, external, complete, source = export_case
    if change == "running":
        write(root / "pipeline_status.json", {"status": "running"})
    elif change == "failed_gate":
        complete["passed"] = False
        write(root / "completion_contract_audit.json", complete)
    elif change == "missing_models":
        complete["expected_tuning_models"] = 39
        write(root / "completion_contract_audit.json", complete)
    elif change == "changed_result":
        write(root / "final/test_summary.csv", {"changed": True})
    elif change == "changed_reference":
        write(external, {"changed": True})
    elif change == "changed_plan":
        write(root / "search_plan.json", {"changed": True})
    elif change == "changed_controller":
        (root / "audit_revision_source/scripts/continue_sac_validation_optimization.py").write_text("# changed controller\n")
    elif change == "changed_finisher":
        (root / "audit_import_revision_source/scripts/finish_sac_fallback_audit_v3.py").write_text("# changed finisher\n")
    elif change == "changed_selection":
        write(root / "selection.json", {"changed_after_freeze": True})
    else:
        value = json.loads((root / "audit_revision.json").read_text())
        value["training_restarted"] = True
        write(root / "audit_revision.json", value)
    with pytest.raises(AssertionError):
        execute(source)


def test_reference_is_rechecked_after_archive(export_case, monkeypatch):
    _, external, _, source = export_case
    original = tarfile.open

    class ChangedOnClose:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self.stream.__enter__()

        def __exit__(self, *args):
            result = self.stream.__exit__(*args)
            write(external, {"changed_after_archive": True})
            return result

    monkeypatch.setattr(tarfile, "open", lambda *a, **kw: ChangedOnClose(original(*a, **kw)))
    with pytest.raises(AssertionError, match="Retained reference evidence changed"):
        execute(source)


def test_export_template_drift_is_rejected():
    with pytest.raises(ValueError, match="boundary changed"):
        collector.optimization_export("not the reviewed exporter", "a" * 64)


def test_collect_restores_shared_template_on_failure(tmp_path, monkeypatch):
    auditor = tmp_path / "audit.py"
    auditor.write_text("# nonfinancial fixture\n")
    monkeypatch.setattr(collector, "AUDITOR_NAME", str(auditor))
    original = collector.transfer.REMOTE_EXPORT

    def failed(*args):
        assert "verify_completed_study" in collector.transfer.REMOTE_EXPORT
        raise RuntimeError("fixture transfer failure")

    monkeypatch.setattr(collector.transfer, "collect", failed)
    with pytest.raises(RuntimeError, match="fixture transfer failure"):
        collector.collect("unused", "/unused", tmp_path / "output", "/unused/python")
    assert collector.transfer.REMOTE_EXPORT == original
