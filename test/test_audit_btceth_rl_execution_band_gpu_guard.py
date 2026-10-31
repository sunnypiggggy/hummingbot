"""Resource-proof fixtures only; no SSH, GPU model, process mutation or prices."""
from copy import deepcopy
import json
import shutil

import pytest

from scripts import audit_btceth_rl_execution_band_gpu_guard as audit
from test.test_audit_btceth_rl_execution_band import protocol_fixture, write_json


def instance():
    return dict(pid=715236,uid=1000,username="sunny",exe="/usr/bin/ptyxis",
                starttime_ticks="10203456",boot_id="1f1e5caa-b676-48ce-9c09-246e1d75001f")


def allowed():
    return [dict(gpu_uuid="GPU-fixture0",pid=715236)]


def snapshot(at):
    return dict(at=at,instance=instance(),argv=audit.SNAPSHOT_QUERY,
        stdout="GPU-fixture0, 715236, /usr/bin/ptyxis, 19 MiB\n",gpu_status_argv=audit.GPU_STATUS_QUERY,
        gpu_status_stdout="0, GPU-fixture0, Tesla V100-SXM2-16GB, 19 MiB, 0 %\n1, GPU-fixture1, Tesla V100-SXM2-16GB, 0 MiB, 0 %\n")


def event():
    return dict(at="2026-10-30T00:00:04Z",argv=audit.EXACT_QUERY,returncode=0,
        stdout_original="GPU-fixture0, 715236\n",stdout_filtered="",instance_observed=instance())


def adapter_fixture(tmp_path):
    roots,dataset=protocol_fixture(tmp_path)
    output,main,reference,source=roots
    original=tmp_path/"original";original.mkdir()
    for name in ("protocol.json","protocol_receipt.json"):
        shutil.copyfile(output/name,original/name)
    frozen=audit.read_json(output/"protocol.json")
    for name in audit.SOURCE_FILES[2:]:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("actual adapter fixture "+name)
    source_hashes={name:audit.sha(source/name) for name in audit.SOURCE_FILES}
    for name in audit.SOURCE_FILES:
        destination=output/"adapter_sources"/name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,destination)
    (original/"validation.log").write_text("Original GPUs are busy; do not stop another task")
    (original/"preflight").mkdir();(original/"preflight/launch.log").write_text("original launch failed1")
    write_json(original/"runner_process.json",dict(status="failed",exit_code=1,
        completed_at="2026-10-30T00:00:01Z",source_sha256=frozen["source_sha256"]["scripts/btceth_rl_execution_band.py"],
        log_sha256=audit.sha(original/"validation.log")))
    fail_hashes={}
    for name in audit.FAILURE_FILES:
        destination=output/"original_failure"/name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original/name,destination)
        fail_hashes[name]=audit.sha(original/name)
    p=dict(prepared_at="2026-10-30T00:00:02Z",original_protocol_sha256=audit.sha(original/"protocol.json"),
        original_protocol_receipt_sha256=audit.sha(original/"protocol_receipt.json"),
        original_failed_files=fail_hashes,source_sha256=source_hashes,allowed_instance=instance(),
        allowed_gpu_rows=allowed(),before=snapshot("2026-10-30T00:00:01.5Z"),
        financial_rules_changed=False,no_annual=True,prices_loaded_at_prepare=False)
    write_json(output/"adapter_protocol.json",p)
    write_json(output/"adapter_protocol_receipt.json",dict(protocol_file="adapter_protocol.json",
        protocol_sha256=audit.sha(output/"adapter_protocol.json"),sealed_at="2026-10-30T00:00:02.1Z"))
    process=dict(command=["python","-u","/research/scripts/btceth_rl_execution_band_gpu_guard.py",
        "--original-runner","/research/scripts/btceth_rl_execution_band.py","--phase","_validate"],
        actual_entrypoint="/research/scripts/btceth_rl_execution_band_gpu_guard.py",
        core_source_sha256=source_hashes[audit.SOURCE_FILES[0]],source_sha256=source_hashes[audit.SOURCE_FILES[0]],
        adapter_source_sha256=source_hashes[audit.SOURCE_FILES[2]],source_adaptation="declared resource adapter",
        started_at="2026-10-30T00:00:03Z",completed_at="2026-10-30T00:00:06Z",child_pid=999)
    write_json(output/"runner_process.json",process)
    runtime=dict(status="completed",old_subprocess_run_restored=True,financial_rules_changed=False,no_annual=True,
        before=snapshot("2026-10-30T00:00:03.5Z"),after=snapshot("2026-10-30T00:00:05Z"),query_events=[event()],
        adapter_protocol_sha256=audit.sha(output/"adapter_protocol.json"),error=None)
    write_json(output/"adapter_runtime.json",runtime)
    return roots,original,dataset


def test_exact_exception_preserves_all_other_pid_rows_and_every_nonexempt_byte():
    text="GPU-fixture0, 12345\nGPU-fixture0, 715236\n GPU-fixture1, 99999\r\n"
    filtered,removed=audit.filtered_query(text,allowed())
    assert filtered=="GPU-fixture0, 12345\n GPU-fixture1, 99999\r\n"
    assert removed==allowed()


def test_only_exact_two_column_rows_can_be_removed():
    text="GPU-fixture0, 715236, added third field\n"
    assert audit.filtered_query(text,allowed())[0]==text
    assert audit.filtered_query("GPU-fixture1, 715236\n",allowed())[0]=="GPU-fixture1, 715236\n"


@pytest.mark.parametrize("key,value",[
    ("pid",123),("uid",0),("username","ubuntu"),("exe","/usr/bin/python"),
    ("starttime_ticks","0"),("boot_id","reused")])
def test_unapproved_or_reused_process_identity_is_rejected(key,value):
    value_instance=instance();value_instance[key]=value
    with pytest.raises(ValueError):audit.validate_instance(value_instance,expected=instance())


@pytest.mark.parametrize("key,value",[("uid",1001),("starttime_ticks","10203457"),
    ("boot_id","2f1e5caa-b676-48ce-9c09-246e1d75001f"),("exe","/opt/ptyxis")])
def test_valid_looking_new_process_is_not_the_authorized_existing_instance(key,value):
    observed=instance();observed[key]=value
    with pytest.raises(ValueError):audit.validate_instance(observed,expected=instance())


@pytest.mark.parametrize("mutation",["query","returncode","reuse","drop_foreign","modified_bytes"])
def test_each_real_filter_event_is_independently_reconstructed(mutation):
    e=event()
    if mutation=="query":e["argv"]=["nvidia-smi"]
    elif mutation=="returncode":e["returncode"]=1
    elif mutation=="reuse":e["instance_observed"]["starttime_ticks"]="10203457"
    elif mutation=="drop_foreign":e["stdout_original"]+="GPU-fixture1, 777\n"
    else:e["stdout_filtered"]=" "
    checks=audit.Checks()
    audit.verify_query_event(e,instance(),allowed(),checks,"event")
    assert checks.failed>0


def test_duplicate_ui_rows_are_not_silently_suppressed():
    with pytest.raises(ValueError):
        audit.filtered_query("GPU-fixture0, 715236\nGPU-fixture0, 715236\n",allowed())


def test_original_failed_attempt_and_transparent_actual_adapter_chain_pass(tmp_path):
    roots,original,_=adapter_fixture(tmp_path)
    checks=audit.Checks();inputs=audit.core.Inputs(*roots,checks)
    proof=audit.verify_adapter(inputs,original)
    assert checks.failed==0,checks.failures
    assert proof["original_failed_attempt_preserved"] and proof["actual_query_events"]==1
    assert proof["financial_rules_changed"] is False


@pytest.mark.parametrize("mutation",[
    "core_source","adapter_source","old_failure","failure_copy","original_protocol","source_snapshot",
    "actual_entrypoint","argv_disguise","restoration","scope","runtime_protocol","resource_foreign",
    "instance_reuse","no_query","wrong_order","annual"])
def test_adapter_cannot_hide_changed_finance_sources_evidence_processes_or_scope(tmp_path,mutation):
    roots,original,_=adapter_fixture(tmp_path)
    output,_,_,source=roots
    process=audit.read_json(output/"runner_process.json")
    runtime=audit.read_json(output/"adapter_runtime.json")
    if mutation=="core_source":(source/audit.SOURCE_FILES[0]).write_text("changed financial core")
    elif mutation=="adapter_source":(source/audit.SOURCE_FILES[2]).write_text("changed adapter")
    elif mutation=="old_failure":(original/"validation.log").write_text("success fabricated")
    elif mutation=="failure_copy":(output/"original_failure/validation.log").write_text("changed copy")
    elif mutation=="original_protocol":(original/"protocol.json").write_text("{}")
    elif mutation=="source_snapshot":(output/"adapter_sources"/audit.SOURCE_FILES[2]).write_text("different source")
    elif mutation=="actual_entrypoint":process["actual_entrypoint"]="/research/scripts/btceth_rl_execution_band.py"
    elif mutation=="argv_disguise":process["command"][2]="/research/scripts/btceth_rl_execution_band.py"
    elif mutation=="restoration":runtime["old_subprocess_run_restored"]=False
    elif mutation=="scope":runtime["financial_rules_changed"]=True
    elif mutation=="runtime_protocol":runtime["adapter_protocol_sha256"]="wrong"
    elif mutation=="resource_foreign":runtime["before"]["stdout"]+="GPU-fixture1, 777, /usr/bin/python, 2000 MiB\n"
    elif mutation=="instance_reuse":runtime["after"]["instance"]["starttime_ticks"]="1111111"
    elif mutation=="no_query":runtime["query_events"]=[]
    elif mutation=="wrong_order":process["started_at"]="2026-10-30T00:00:01Z"
    else:(output/"test_started.json").write_text("{}")
    write_json(output/"runner_process.json",process);write_json(output/"adapter_runtime.json",runtime)
    checks=audit.Checks();inputs=audit.core.Inputs(*roots,checks)
    try:audit.verify_adapter(inputs,original)
    except (ValueError,KeyError):pass
    assert checks.failed>0


def test_audit_artifact_writer_never_overwrites_previous_failure_or_pass(tmp_path):
    path=tmp_path/"audit.json"
    audit.dump_new(path,{"passed":False})
    with pytest.raises(FileExistsError):audit.dump_new(path,{"passed":True})
    assert json.loads(path.read_text())=={"passed":False}


@pytest.mark.parametrize("financial_passed",[True,False])
def test_wrapper_adds_proofs_without_replacing_or_weakening_the_original_financial_audit(tmp_path,monkeypatch,financial_passed):
    roots,original,dataset=adapter_fixture(tmp_path)
    output,main,reference,source=roots
    calls=[]
    def original_audit(*args,**kwargs):
        calls.append((args,kwargs))
        return dict(passed=financial_passed,status="passed" if financial_passed else "failed",
            run_count=12,comparator_run_count=12,repeat_count=12,test_complete=False,
            validation_complete=financial_passed,checks_total=1000,failure_count=0 if financial_passed else 1,
            failures=[] if financial_passed else ["Original ledger rejected"],input_sha256={key:{} for key in
                ("extension","main","reference","source")})
    monkeypatch.setattr(audit.core,"audit",original_audit)
    result=audit.audit(output,original,main,reference,dataset,source_root=source)
    assert len(calls)==1
    assert result["core_financial_audit"]["checks_total"]==1000
    assert result["checks_total"]>1000
    assert result["passed"] is financial_passed
    assert result["resource_adapter_disclosed"]
    assert result["adapter_evidence"]["original_failed_attempt_preserved"]
    if not financial_passed:
        assert "Original ledger rejected" in result["failures"]


def test_invalid_resource_proofs_prevent_financial_reconstruction(tmp_path,monkeypatch):
    roots,original,dataset=adapter_fixture(tmp_path)
    output,main,reference,source=roots
    (output/"test_started.json").write_text("{}")
    calls=[]
    monkeypatch.setattr(audit.core,"audit",lambda *a,**k:calls.append(True))
    result=audit.audit(output,original,main,reference,dataset,source_root=source)
    assert result["passed"] is False and not calls and result["run_count"]==0
