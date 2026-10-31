"""Comparison arithmetic and presentation lineage, never formal fake report data."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

from scripts import sac_validation_optimization_report_app as report


def arithmetic_accounts():
    rows=[]
    for symbol in report.SYMBOLS:
        for seed in report.SEEDS:
            for cost in report.core.COSTS:
                raw={"run_id":f"test__{cost}__{symbol}__sac__{seed}","symbol":symbol,"algorithm":"sac","seed":seed,"scenario":cost,"split":"test","net_return":(seed-15)/100,"max_drawdown":-seed/100}
                for variant in report.VARIANTS:
                    row=report.account_identity(raw,variant)
                    if variant=='after':row['net_return']+=.01;row['max_drawdown']-=.02
                    rows.append(row)
    return rows


def test_policy_namespaces_do_not_change_native_algorithm_or_source_run():
    raw={"run_id":"test__base__BTCUSDT__sac__0","symbol":"BTCUSDT","algorithm":"sac","seed":0,"scenario":"base","split":"test","net_return":.18,"max_drawdown":-.14}
    before=deepcopy(raw)
    a=report.account_identity(raw,'before');b=report.account_identity(raw,'after')
    assert raw==before
    assert a['run_id']==b['run_id']==a['source_run_id']==b['source_run_id']==raw['run_id']
    assert a['algorithm']==b['algorithm']=='sac'
    assert a['account_id']!=b['account_id']


@pytest.mark.parametrize('bad',[{'symbol':'ADAUSDT'},{'seed':100},{'algorithm':'sac_optimized'},{'split':'validation'},{'scenario':'free'}])
def test_foreign_or_tuning_accounts_cannot_enter_annual_comparison(bad):
    raw={"run_id":"identity-only","symbol":"BTCUSDT","algorithm":"sac","seed":0,"scenario":"base","split":"test"};raw.update(bad)
    with pytest.raises(ValueError):report.account_identity(raw,'after')


def test_statistics_and_matched_differences_use_all_thirty_seeds():
    accounts=arithmetic_accounts();stats=report.seed_statistics(accounts);pairs=report.matched_seed_results(accounts);paired=report.matched_statistics(pairs)
    assert len(accounts)==240 and len(stats)==8 and len(pairs)==120 and len(paired)==4
    original=next(r for r in stats if r['symbol']=='BTCUSDT' and r['policy_variant']=='before' and r['scenario']=='base')
    assert (original['positive_seeds'],original['loss_seeds'],original['zero_seeds'])==(14,15,1)
    assert [original[k] for k in ('p10_return','q1_return','median_return','q3_return','p90_return')]==pytest.approx([-.121,-.0775,-.005,.0675,.111])
    assert original['median_drawdown']==pytest.approx(.145) and original['worst_drawdown']==pytest.approx(.29)
    assert all(r['seeds']==30 and r['median_return_difference']==pytest.approx(.01) and r['median_drawdown_difference']==pytest.approx(.02) for r in paired)
    assert all(r['higher_return_seeds']==30 and r['lower_drawdown_seeds']==r['both_better_seeds']==r['unchanged_seeds']==0 for r in paired)


@pytest.mark.parametrize('change',['remove_seed','duplicate_seed','nonfinite'])
def test_incomplete_or_bad_seed_distribution_rejected(change):
    rows=arithmetic_accounts()
    if change=='remove_seed':rows.pop()
    elif change=='duplicate_seed':rows.append(deepcopy(rows[0]))
    else:rows[-1]['net_return']=float('nan')
    with pytest.raises(ValueError):report.seed_statistics(rows)


def test_fallback_same_results_have_zero_matched_difference():
    rows=arithmetic_accounts()
    originals={(r['symbol'],r['seed'],r['scenario']):r for r in rows if r['policy_variant']=='before'}
    for row in rows:
        if row['policy_variant']=='after':
            source=originals[(row['symbol'],row['seed'],row['scenario'])]
            row['net_return'],row['max_drawdown']=source['net_return'],source['max_drawdown']
    results=report.matched_statistics(report.matched_seed_results(rows))
    assert all(r['median_return_difference']==r['median_drawdown_difference']==0 for r in results)
    assert all(r['unchanged_seeds']==30 and r['higher_return_seeds']==r['lower_drawdown_seeds']==r['both_better_seeds']==0 for r in results)


def test_only_canonical_root_relative_evidence_paths(tmp_path):
    for name in ('../escape','/absolute','C:/drive','a\\b','a/./b',''):
        with pytest.raises(ValueError):report.relative(tmp_path,name)


def test_actual_prepared_source_contract_rejects_incomplete_study_before_output():
    """Real prepared protocol must reach missing completion, never create HTML."""
    top=report.ROOT/'results/backtests/sac_validation_optimization_20261031'
    if not (top/'extension_manifest.json').is_file():pytest.skip('Actual optimization study not prepared locally')
    extension=report.read(top/'extension_manifest.json');spec=report.read(top/'protocol.json');plan=report.read(top/'search_plan.json')
    assert report.sha(top/'search_plan.json')==report.PLAN_SHA256
    assert spec['symbols']==list(report.SYMBOLS) and spec['tuning_seeds']==list(report.TUNING_SEEDS) and spec['final_seeds']==list(report.SEEDS)
    assert spec['candidate_order']==list(report.CANDIDATES) and spec['candidate_recipes']=={r['candidate_id']:r['native_recipe'] for r in plan['candidates']}
    assert spec['capital_per_account']==10000 and spec['capital_shared_between_pairs'] is False
    for name,digest in {**spec['sources'],**extension['extension_sources']}.items():
        prefix='frozen_source/' if name in spec['sources'] else 'extension_source/'
        assert report.sha(top/(prefix+name))==report.sha(report.ROOT/name)==digest
    if (top/'completion_contract_audit.json').is_file():pytest.skip('Complete study now requires final numerical QA instead of partial rejection')
    output_before=(top/'report').exists()
    missing=next(name for name in ('audit_revision.json','audit_import_revision.json','completion_contract_audit.json') if not (top/name).is_file())
    with pytest.raises(ValueError,match='Missing evidence: '+missing):
        report.make_reviewed_snapshot(top,baseline_btc_root=report.ROOT/'results/backtests/btc_rl_seed_stability_20261031',baseline_eth_root=report.ROOT/'results/backtests/eth_rl_seed_stability_20261031',retained_study_root=report.ROOT/'results/backtests/single_pair_rl_six_20261030')
    assert (top/'report').exists()==output_before


def export_receipt_fixture(top):
    """Tiny archive/receipt metadata only; no synthetic prices or model outcomes."""
    top.mkdir(exist_ok=True)
    dataset='1'*64;manifest='2'*64;auditor='3'*64
    source_roots={s:{'source_root':'/research/original/'+s} for s in report.SYMBOLS}
    report.write(top/'protocol.json',{'dataset_sha256':dataset,'dataset_manifest_sha256':manifest,'reference_sources':source_roots})
    external={'/research/data/dataset.pkl.gz':dataset,'/research/data/manifest.json':manifest}
    maps=[]
    for suffix in ('training/checkpoint/model.zip','evaluation/test/metadata.json'):
        files={}
        for symbol in report.SYMBOLS:
            name=f'references/{symbol}/{suffix}';path=top/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'\x00\xffhash-only transport unit evidence\x01')
            files[name]=report.sha(path);external[source_roots[symbol]['source_root']+'/'+suffix]=report.sha(path)
        maps.append(files)
    report.write(top/'reference_manifest.json',{'files_sha256':maps[0]})
    report.write(top/'reference_annual_import.json',{'files_sha256':maps[1]})
    completion={'external_inputs_sha256':external}
    report.write(top/'completion_contract_audit.json',completion)
    report.write(top/'pipeline_launch_revision.json',{'metadata_only_unit_export_member':True})
    report.write(top/'pipeline_launch_import_revision.json',{'metadata_only_unit_final_export_member':True})
    files={name:report.sha(top/name) for name in ['protocol.json','completion_contract_audit.json','reference_manifest.json','reference_annual_import.json','pipeline_launch_revision.json','pipeline_launch_import_revision.json',*maps[0],*maps[1]]}
    transport=top/'transport';transport.mkdir()
    archive=transport/'unit_metadata_audited.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:
        for name in files:stream.add(top/name,arcname=name,recursive=False)
    receipt={'completion_contract_audit_sha256':report.sha(top/'completion_contract_audit.json'),
        'external_inputs_sha256':external,'external_verified_sha256':external,'external_rechecked_after_archive':True,
        'verifier_source_sha256':auditor,'audit_revision_sha256':'5'*64,'audit_import_revision_sha256':'8'*64,'archive':'/remote/unit_metadata_audited.tar.gz','archive_sha256':report.sha(archive),'files':files}
    path=transport/'unit_metadata_audited.tar.gz.receipt.json';report.write(path,receipt)
    return completion,auditor,path


def test_external_linux_dataset_is_bound_by_verified_transport_not_loaded(tmp_path):
    completion,extension,_=export_receipt_fixture(tmp_path);pins={}
    result=report.verify_transport_external(tmp_path,completion,extension,pins)
    assert result['external_inputs_sha256']==completion['external_inputs_sha256']
    assert result['external_rechecked_after_archive'] is True
    assert len(pins)==2


@pytest.mark.parametrize('bad',['no_recheck','wrong_external','wrong_auditor','archive_changed','local_file_changed','completion_changed','reference_external_missing','reference_external_changed','unexpected_external'])
def test_transport_external_or_local_tamper_rejected(tmp_path,bad):
    completion,extension,path=export_receipt_fixture(tmp_path);receipt=report.read(path)
    if bad=='no_recheck':receipt['external_rechecked_after_archive']=False
    elif bad=='wrong_external':receipt['external_verified_sha256']={}
    elif bad=='wrong_auditor':receipt['verifier_source_sha256']='4'*64
    elif bad=='archive_changed':(tmp_path/'transport/unit_metadata_audited.tar.gz').write_bytes(b'altered archive')
    elif bad=='local_file_changed':(tmp_path/'protocol.json').write_text('{}',encoding='utf-8')
    elif bad=='completion_changed':(tmp_path/'completion_contract_audit.json').write_text('{}',encoding='utf-8')
    elif bad=='reference_external_missing':del completion['external_inputs_sha256']['/research/original/BTCUSDT/training/checkpoint/model.zip']
    elif bad=='reference_external_changed':completion['external_inputs_sha256']['/research/original/BTCUSDT/training/checkpoint/model.zip']='a'*64
    else:completion['external_inputs_sha256']['/research/unexpected']='b'*64
    if bad not in ['archive_changed','local_file_changed','completion_changed']:report.write(path,receipt)
    with pytest.raises(ValueError):report.verify_transport_external(tmp_path,completion,extension,{})


@pytest.mark.parametrize('changed',[None,'revision','launch'])
def test_audited_export_binds_actual_revision_and_continuation_launch(tmp_path,changed):
    completion,auditor,path=export_receipt_fixture(tmp_path);receipt=report.read(path)
    launch_sha=report.sha(tmp_path/'pipeline_launch_revision.json')
    if changed=='revision':receipt['audit_revision_sha256']='6'*64
    if changed=='launch':receipt['files']['pipeline_launch_revision.json']='7'*64
    report.write(path,receipt)
    kwargs={'audit_revision_sha256':'5'*64,'pipeline_launch_revision_sha256':launch_sha}
    if changed:
        with pytest.raises(ValueError):report.verify_transport_external(tmp_path,completion,auditor,{},**kwargs)
    else:assert report.verify_transport_external(tmp_path,completion,auditor,{},**kwargs)['verifier_source_sha256']==auditor


@pytest.mark.parametrize('changed',[None,'revision','launch'])
def test_final_export_cannot_substitute_import_revision_or_finisher_launch(tmp_path,changed):
    completion,auditor,path=export_receipt_fixture(tmp_path);receipt=report.read(path)
    launch_sha=report.sha(tmp_path/'pipeline_launch_import_revision.json')
    if changed=='revision':receipt['audit_import_revision_sha256']='9'*64
    if changed=='launch':receipt['files']['pipeline_launch_import_revision.json']='a'*64
    report.write(path,receipt)
    kwargs={'audit_import_revision_sha256':'8'*64,'pipeline_launch_import_revision_sha256':launch_sha}
    if changed:
        with pytest.raises(ValueError):report.verify_transport_external(tmp_path,completion,auditor,{},**kwargs)
    else:assert report.verify_transport_external(tmp_path,completion,auditor,{},**kwargs)['verifier_source_sha256']==auditor


@pytest.mark.parametrize('passed',[False,'true',1])
def test_audits_require_true_bool_not_truthy(tmp_path,passed):
    report.write(tmp_path/'receipt.json',{'passed':passed,'failure_count':0,'failures':[],'checks_total':1})
    with pytest.raises(ValueError):report.bind_audit(tmp_path,'receipt.json',{})


def revision_fixture(top,monkeypatch):
    """Only source/version metadata; no financial case or model is fabricated."""
    repo=top/'repo';study=top/'study';repo.mkdir();study.mkdir()
    original_plan=report.ROOT/'results/backtests/sac_validation_optimization_20261031_preflight/search_plan.json'
    shutil.copyfile(original_plan,study/'search_plan.json')
    extension={'extension_sources':{report.EXTENSION_FILES[2]:report.ORIGINAL_AUDITOR_SHA256}}
    report.write(study/'extension_manifest.json',extension)
    report.write(study/'protocol.json',{'frozen_at':'2026-10-31T04:03:12+00:00','sources':{'scripts/audit_single_pair_rl.py':'4'*64}})
    report.write(study/'pipeline_launch.json',{'pid':123,'metadata_only_unit_fixture':True})
    digests=[]
    for name in report.REVISION_FILES:
        source=repo/name;source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes((report.ROOT/name).read_bytes())
        copied=study/('audit_revision_source/'+name);copied.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,copied);digests.append(report.sha(source))
    revision={'schema_version':1,'revision_id':'native_sac_policy_kwargs_default_false_v2','reason_code':'sb3_sac_saved_policy_kwargs_use_sde_false',
        'created_at_utc':'2026-10-31T04:30:00+00:00','original_protocol_sha256':report.sha(study/'protocol.json'),'search_plan_sha256':report.PLAN_SHA256,
        'original_extension_manifest_sha256':report.sha(study/'extension_manifest.json'),'original_audit_source_sha256':report.ORIGINAL_AUDITOR_SHA256,
        'revised_audit_source_path':'audit_revision_source/'+report.REVISION_FILES[0],'revised_audit_source_sha256':digests[0],
        'controller_source_path':'audit_revision_source/'+report.REVISION_FILES[1],'controller_source_sha256':digests[1],
        'original_pipeline_launch_sha256':report.sha(study/'pipeline_launch.json'),'financial_oracle_source_sha256':'4'*64,
        'permitted_metadata_normalization':report.REVISION_NORMALIZATION,'financial_math_changed':False,'model_recipe_changed':False,'training_restarted':False}
    report.write(study/'audit_revision.json',revision)
    report.write(study/'audit_revision_receipt.json',{'sha256':report.sha(study/'audit_revision.json'),'at_utc':'2026-10-31T04:30:01+00:00'})
    monkeypatch.setattr(report,'ROOT',repo)
    return study,extension,revision


def test_audit_revision_retains_original_and_identifies_two_active_sources(tmp_path,monkeypatch):
    top,extension,revision=revision_fixture(tmp_path,monkeypatch);pins={}
    assert report.verify_audit_revision(top,extension,pins)==revision
    assert set(pins)=={'audit_revision.json','audit_revision_receipt.json','pipeline_launch.json',*('audit_revision_source/'+n for n in report.REVISION_FILES)}
    assert extension['extension_sources'][report.EXTENSION_FILES[2]]==report.ORIGINAL_AUDITOR_SHA256


@pytest.mark.parametrize('changed',[None,'audit_revision_sha256','original_pipeline_launch_sha256','controller_source_sha256','revised_audit_source_sha256','training_restarted','numeric_false'])
def test_continuation_launch_binds_revision_sources_and_original_launch(tmp_path,monkeypatch,changed):
    top,extension,revision=revision_fixture(tmp_path,monkeypatch);pins={};report.verify_audit_revision(top,extension,pins)
    launch={'audit_revision_sha256':pins['audit_revision.json'],'original_pipeline_launch_sha256':revision['original_pipeline_launch_sha256'],
        'controller_source_sha256':revision['controller_source_sha256'],'revised_audit_source_sha256':revision['revised_audit_source_sha256'],'training_restarted':False}
    if changed=='numeric_false':launch['training_restarted']=0
    elif changed=='training_restarted':launch[changed]=True
    elif changed:launch[changed]='changed'
    report.write(top/'pipeline_launch_revision.json',launch)
    if changed:
        with pytest.raises(ValueError):report.verify_revision_launch(top,revision,pins)
    else:
        assert report.verify_revision_launch(top,revision,pins)==launch
        assert pins['pipeline_launch_revision.json']==report.sha(top/'pipeline_launch_revision.json')


@pytest.mark.parametrize('changed',['original_protocol_sha256','search_plan_sha256','original_extension_manifest_sha256','original_audit_source_sha256','financial_oracle_source_sha256',
    'revised_audit_source_sha256','controller_source_sha256','original_pipeline_launch_sha256','permitted_metadata_normalization','financial_math_changed','model_recipe_changed','training_restarted','numeric_false','active_source_changed','controller_copy_changed'])
def test_audit_revision_cannot_change_original_or_claim_legacy_auditor(tmp_path,monkeypatch,changed):
    top,extension,revision=revision_fixture(tmp_path,monkeypatch)
    if changed=='active_source_changed':(report.ROOT/report.REVISION_FILES[0]).write_bytes(b'changed')
    elif changed=='controller_copy_changed':(top/('audit_revision_source/'+report.REVISION_FILES[1])).write_bytes(b'changed')
    elif changed=='numeric_false':revision['training_restarted']=0
    elif changed in ('financial_math_changed','model_recipe_changed','training_restarted'):revision[changed]=True
    else:revision[changed]='changed'
    report.write(top/'audit_revision.json',revision)
    report.write(top/'audit_revision_receipt.json',{'sha256':report.sha(top/'audit_revision.json'),'at_utc':'2026-10-31T04:30:01+00:00'})
    with pytest.raises(ValueError):report.verify_audit_revision(top,extension,{})


@pytest.mark.parametrize('changed',['sha256','at_utc'])
def test_audit_revision_sidecar_hash_and_chronology_required(tmp_path,monkeypatch,changed):
    top,extension,_=revision_fixture(tmp_path,monkeypatch);receipt=report.read(top/'audit_revision_receipt.json')
    receipt[changed]='0'*64 if changed=='sha256' else '2026-10-31T04:29:59+00:00'
    report.write(top/'audit_revision_receipt.json',receipt)
    with pytest.raises(ValueError):report.verify_audit_revision(top,extension,{})


def import_revision_fixture(tmp_path,monkeypatch):
    """Version-chain identities only; no NAVs, prices, financial case or HTML."""
    top,extension,parent=revision_fixture(tmp_path,monkeypatch);pins={};report.verify_audit_revision(top,extension,pins)
    report.write(top/'pipeline_launch_revision.json',{'audit_revision_sha256':pins['audit_revision.json'],'original_pipeline_launch_sha256':parent['original_pipeline_launch_sha256'],
        'controller_source_sha256':parent['controller_source_sha256'],'revised_audit_source_sha256':parent['revised_audit_source_sha256'],'training_restarted':False})
    report.verify_revision_launch(top,parent,pins)
    report.write(top/'selection.json',{'at':'2026-10-31T04:40:00+00:00','selection':{s:{'candidate_id':'c00_baseline','fallback_to_baseline':True} for s in report.SYMBOLS}})
    report.write(top/'selection_audit.json',{'passed':True,'failure_count':0,'audit_source_sha256':report.PARENT_AUDITOR_SHA256,'selection_sha256':report.sha(top/'selection.json')})
    report.write(top/'final/validation_optimization_audit.json',{'passed':False,'failure_count':1,'core_financial_audit_passed':False,'audited_at':'2026-10-31T04:50:00+00:00','phase':'final-validation','audit_source_sha256':report.PARENT_AUDITOR_SHA256,'protocol_sha256':report.sha(top/'protocol.json')})
    digests=[]
    for name in report.IMPORT_REVISION_FILES:
        source=report.ROOT/name;source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(('unit v3 source pin '+name).encode())
        copied=top/('audit_import_revision_source/'+name);copied.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,copied);digests.append(report.sha(source))
    revision={'schema_version':1,'revision_id':'retained_import_population_shadowing_v3','reason_code':'import_manifest_expected_digest_shadowing','created_at_utc':'2026-10-31T05:00:00+00:00',
        'original_protocol_sha256':report.sha(top/'protocol.json'),'search_plan_sha256':report.PLAN_SHA256,'parent_audit_revision_sha256':pins['audit_revision.json'],'parent_auditor_source_sha256':report.PARENT_AUDITOR_SHA256,
        'revised_audit_source_path':'audit_import_revision_source/'+report.IMPORT_REVISION_FILES[0],'revised_audit_source_sha256':digests[0],
        'finisher_source_path':'audit_import_revision_source/'+report.IMPORT_REVISION_FILES[1],'finisher_source_sha256':digests[1],
        'failed_final_validation_audit_sha256':report.sha(top/'final/validation_optimization_audit.json'),'selection_sha256':report.sha(top/'selection.json'),'selection_audit_sha256':report.sha(top/'selection_audit.json'),
        'permitted_change':report.IMPORT_REVISION_CHANGE,'financial_math_changed':False,'model_recipe_changed':False,'selection_changed':False,'training_restarted':False}
    report.write(top/'audit_import_revision.json',revision)
    report.write(top/'audit_import_revision_receipt.json',{'sha256':report.sha(top/'audit_import_revision.json'),'at_utc':'2026-10-31T05:00:01+00:00'})
    return top,parent,revision,pins


def test_final_import_revision_preserves_failed_parent_and_immutable_selection(tmp_path,monkeypatch):
    top,parent,revision,pins=import_revision_fixture(tmp_path,monkeypatch)
    assert report.verify_import_audit_revision(top,parent,pins)==revision
    assert report.read(top/'final/validation_optimization_audit.json')['passed'] is False
    assert all(name in pins for name in ('audit_import_revision.json','audit_import_revision_receipt.json','selection.json','selection_audit.json','final/validation_optimization_audit.json',*('audit_import_revision_source/'+n for n in report.IMPORT_REVISION_FILES)))


@pytest.mark.parametrize('changed',['parent_audit_revision_sha256','parent_auditor_source_sha256','original_protocol_sha256','search_plan_sha256','failed_final_validation_audit_sha256','selection_sha256','selection_audit_sha256',
    'revised_audit_source_sha256','finisher_source_sha256','financial_math_changed','model_recipe_changed','selection_changed','training_restarted','failed_wrapper_claimed_passed','new_selection','stale_sidecar','active_source_changed'])
def test_final_import_revision_cannot_replace_old_evidence_or_training(tmp_path,monkeypatch,changed):
    top,parent,revision,pins=import_revision_fixture(tmp_path,monkeypatch)
    if changed=='failed_wrapper_claimed_passed':
        failed=report.read(top/'final/validation_optimization_audit.json');failed['passed']=True;report.write(top/'final/validation_optimization_audit.json',failed)
        revision['failed_final_validation_audit_sha256']=report.sha(top/'final/validation_optimization_audit.json')
    elif changed=='new_selection':
        choice=report.read(top/'selection.json');choice['selection']['BTCUSDT']['candidate_id']='c01_lr1e4';report.write(top/'selection.json',choice)
        revision['selection_sha256']=report.sha(top/'selection.json')
    elif changed=='active_source_changed':(report.ROOT/report.IMPORT_REVISION_FILES[0]).write_bytes(b'changed')
    elif changed in ('financial_math_changed','model_recipe_changed','selection_changed','training_restarted'):revision[changed]=True
    elif changed!='stale_sidecar':revision[changed]='changed'
    report.write(top/'audit_import_revision.json',revision)
    if changed!='stale_sidecar':report.write(top/'audit_import_revision_receipt.json',{'sha256':report.sha(top/'audit_import_revision.json'),'at_utc':'2026-10-31T05:00:01+00:00'})
    else:report.write(top/'audit_import_revision_receipt.json',{'sha256':'0'*64,'at_utc':'2026-10-31T05:00:01+00:00'})
    with pytest.raises(ValueError):report.verify_import_audit_revision(top,parent,pins)


@pytest.mark.parametrize('changed',[None,'audit_import_revision_sha256','parent_pipeline_launch_revision_sha256','finisher_source_sha256','revised_audit_source_sha256','training_restarted'])
def test_final_fallback_launch_binds_both_revisions_and_sources(tmp_path,monkeypatch,changed):
    top,parent,revision,pins=import_revision_fixture(tmp_path,monkeypatch);report.verify_import_audit_revision(top,parent,pins)
    launch={'audit_import_revision_sha256':pins['audit_import_revision.json'],'parent_pipeline_launch_revision_sha256':pins['pipeline_launch_revision.json'],
        'finisher_source_sha256':revision['finisher_source_sha256'],'revised_audit_source_sha256':revision['revised_audit_source_sha256'],'training_restarted':False}
    if changed:launch[changed]=True if changed=='training_restarted' else 'changed'
    report.write(top/'pipeline_launch_import_revision.json',launch)
    if changed:
        with pytest.raises(ValueError):report.verify_import_revision_launch(top,revision,pins)
    else:assert report.verify_import_revision_launch(top,revision,pins)==launch


@pytest.mark.parametrize('changed',['legacy_source','no_revision','protocol'])
def test_passed_active_audit_must_bind_version_receipt_and_original_protocol(tmp_path,changed):
    report.write(tmp_path/'audit_revision.json',{'unit_version_metadata':True});digest=report.sha(tmp_path/'audit_revision.json')
    receipt={'passed':True,'failure_count':0,'failures':[],'checks_total':1,'audit_source_sha256':'a'*64,'protocol_sha256':'b'*64,'inputs_sha256':{'audit_revision.json':digest}}
    if changed=='legacy_source':receipt['audit_source_sha256']=report.ORIGINAL_AUDITOR_SHA256
    elif changed=='no_revision':receipt['inputs_sha256']={}
    else:receipt['protocol_sha256']='c'*64
    report.write(tmp_path/'passed.json',receipt)
    with pytest.raises(ValueError):report.bind_audit(tmp_path,'passed.json',{},active_auditor_sha256='a'*64,audit_revision_sha256=digest,protocol_sha256='b'*64)


@pytest.mark.parametrize('changed',[None,'v2_source','missing_revision','missing_sidecar'])
def test_final_passed_receipt_uses_v3_and_preserves_old_failed_path(tmp_path,changed):
    assert report.FINAL_AUDIT_PATHS=={'validation':'final/validation_optimization_audit_v3.json','test':'final/test_optimization_audit_v3.json'}
    report.write(tmp_path/'audit_import_revision.json',{'unit_revision_identity':True});report.write(tmp_path/'audit_import_revision_receipt.json',{'unit_sidecar_identity':True})
    revision_sha=report.sha(tmp_path/'audit_import_revision.json');sidecar_sha=report.sha(tmp_path/'audit_import_revision_receipt.json')
    value={'passed':True,'failure_count':0,'failures':[],'checks_total':1,'phase':'final-validation','audit_source_sha256':'9'*64,
        'inputs_sha256':{'audit_import_revision.json':revision_sha,'audit_import_revision_receipt.json':sidecar_sha}}
    if changed=='v2_source':value['audit_source_sha256']=report.PARENT_AUDITOR_SHA256
    if changed=='missing_revision':del value['inputs_sha256']['audit_import_revision.json']
    if changed=='missing_sidecar':del value['inputs_sha256']['audit_import_revision_receipt.json']
    report.write(tmp_path/'final/validation_optimization_audit_v3.json',value)
    kwargs={'phase':'final-validation','active_auditor_sha256':'9'*64,'audit_import_revision_sha256':revision_sha,'audit_import_revision_receipt_sha256':sidecar_sha}
    if changed:
        with pytest.raises(ValueError):report.bind_audit(tmp_path,report.FINAL_AUDIT_PATHS['validation'],{},**kwargs)
    else:assert report.bind_audit(tmp_path,report.FINAL_AUDIT_PATHS['validation'],{},**kwargs)['passed'] is True


def test_complete_candidate_gate_rows_keep_baseline_reference_raw_values():
    keys=['score_improvement_at_least','paired_better_seed_count_at_least','positive_base_seed_count_at_least','stress_return_median_at_least','stress_p90_drawdown_above_baseline_at_most','nonworse_subwindows_at_least']
    metric={'robust_score':.1,'seed_scores':[.1]*5,'base_positive_seeds':3,'stress_median_return':0.,'stress_p90_drawdown':.2,'subwindow_median_scores':[.1]*3}
    choice={'metrics_by_candidate':{cid:deepcopy(metric) for cid in report.CANDIDATES},'gates_by_candidate':{cid:{k:False for k in keys} for cid in report.CANDIDATES}}
    selection={'selection':{s:deepcopy(choice) for s in report.SYMBOLS}};before=deepcopy(selection)
    plan={'acceptance':dict(zip(keys,[.01,3,3,0,.01,2]))}
    rows=report.gate_rows(selection,plan)
    assert len(rows)==48 and selection==before
    assert sum(not r['applicable'] for r in rows)==12
    assert all(r['passed'] is False for r in rows)
    assert all(r['actual']==0 for r in rows if r['candidate_id']=='c00_baseline' and r['gate']=='score_improvement_at_least')


def test_official_compiler_actual_original_rows_and_explicit_display_contract():
    """Real source results exercise authored props, without building a report.

    The after namespace is an explicit identical-account fallback fixture only;
    this test neither fabricates optimized outcomes nor asserts browser rendering.
    """
    root=Path(__file__).resolve().parents[1]
    original=root/'results/backtests/btc_rl_seed_stability_20261031/report_v2/reviewed.json'
    if not original.is_file() or shutil.which('wsl.exe') is None:
        pytest.skip('Real original reviewed accounts and WSL official compiler required')
    script=r'''
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {loadPrebuiltCompiler} from 'file:///mnt/c/Users/sunny/.codex/plugins/cache/openai-curated-remote/data-analytics/1.1.1-exp-components.2/output_formats/html/scripts/data-app-runtime.mjs';
const original=JSON.parse(await readFile('/mnt/h/PycharmProjects/hummingbot/results/backtests/btc_rl_seed_stability_20261031/report_v2/reviewed.json','utf8'));
const beforeJSON=JSON.stringify(original),source=await readFile('/mnt/h/PycharmProjects/hummingbot/scripts/sac_validation_optimization_report_app.jsx','utf8');
const compiler=await loadPrebuiltCompiler(),compiled=compiler.transform(source,{filePath:'ReportContent.jsx'});
compiler.parseJavaScript(compiled,{sourceType:'script'});
compiler.parseCss(await readFile('/mnt/h/PycharmProjects/hummingbot/scripts/sac_validation_optimization_report_app.css','utf8'));
const originalRows=id=>original.queries[id].rows;
const qualify=(rows,variant)=>rows.filter(r=>r.algorithm==='sac').map(r=>({...r,policy_variant:variant,source_run_id:r.run_id,account_id:`${variant}::${r.run_id}`,model_origin:variant==='after'?'retained_baseline_fallback':'original_baseline'}));
const queries={};
for(const id of ['account_summary','equity_history','monthly_returns'])queries[id]=['before','after'].flatMap(v=>qualify(originalRows(id),v));
queries.account_summary.push(...originalRows('account_summary').filter(r=>r.seed===null).map(r=>({...r,policy_variant:'benchmark',source_run_id:r.run_id,account_id:`benchmark::${r.run_id}`})));
queries.seed_statistics=originalRows('cohort_statistics').filter(r=>r.algorithm==='sac'&&r.cohort==='all30').flatMap(r=>['before','after'].map(v=>({...r,policy_variant:v})));
queries.training_records=originalRows('training_records').filter(r=>r.algorithm==='sac').flatMap(r=>['before','after'].map(v=>({...r,model_id:`${v}:${r.symbol}:${r.seed}`,stage:v==='before'?'baseline':'final',policy_variant:v,candidate_id:'c00_baseline',model_origin:v==='before'?'original_baseline':'retained_baseline_fallback'})));
queries.selection=['BTCUSDT','ETHUSDT'].map(symbol=>({symbol,candidate_id:'c00_baseline',fallback_to_baseline:true,learning_rate:.0003,gamma:.995,frozen_at:'unit-contract-only'}));
queries.validation_gates=['BTCUSDT','ETHUSDT'].map(symbol=>({gate_key:symbol+':baseline-reference',symbol,candidate_id:'c00_baseline',passed:false,applicable:false}));
queries.protocol=[{before_after_model_records:120,tuning_models:40,new_training_models:40,fresh_final_models:0,reused_final_models:60}];
for(const id of ['matched_seed_results','matched_statistics','validation_candidates','audit_evidence','reference_provenance'])queries[id]=[];
const jsx=(type,props)=>({type,props:props??{}}),module={exports:{}};
const publicComponents=Object.fromEntries(['DataComponent','DataTable','Dropdown','EvidenceChart','ReportSection','RichNarrative'].map(n=>[n,n]));
publicComponents.useDataApp=()=>({appTitle:'Original-account fallback prop test only',setAppTitle:()=>{},canEdit:false,mode:'view',reviewedPeriodRows:id=>{assert.ok(Object.hasOwn(queries,id),id);return queries[id];}});
vm.runInNewContext(compiled,{module,exports:module.exports,require:name=>{
 if(name==='react')return {useMemo:f=>f(),useState:v=>[v,()=>{}]};
 if(name==='react/jsx-runtime')return {jsx,jsxs:jsx,Fragment:'Fragment'};
 if(name==='../../data-app-public.jsx')return publicComponents;
 if(name.endsWith('.css'))return {};
 throw Error('Unexpected authored import '+name);
}});
const training=module.exports.trainingDisplayRows(queries.training_records);
assert.equal(training.length,120);assert.equal(new Set(training.map(r=>r.key)).size,120);
for(let i=0;i<training.length;i++){
 for(const key of Object.keys(queries.training_records[i]))assert.equal(JSON.stringify(training[i][key]),JSON.stringify(queries.training_records[i][key]));
 assert.equal(training[i].stage_label,training[i].policy_variant==='before'?'原模型':'回退复用原模型');
}
assert.equal(module.exports.accountPairs(queries.account_summary).length,128);
const tables=[],charts=[];
function validate(props){
 assert.ok(Object.hasOwn(props,'displayRows')&&Array.isArray(props.displayRows),'explicit displayRows required');
 assert.equal(props.children.type,'DataTable');assert.equal(props.children.props.rows,props.displayRows,'exact shared mapped row array required');
 assert.ok(Array.isArray(props.sourceRows));assert.ok(Object.hasOwn(queries,props.queryId));
 for(const r of props.sourceRows)assert.ok(queries[props.queryId].includes(r),'source rows must come from the declared reviewed query');
 const keys=props.displayRows.map(r=>r[props.children.props.rowKey]);assert.equal(new Set(keys).size,keys.length,'table identity must be unique');
}
function walk(n){
 if(Array.isArray(n)){n.forEach(walk);return;}if(!n||typeof n!=='object')return;
 if(typeof n.type==='function'){walk(n.type(n.props));return;}
 if(n.type==='DataComponent'){validate(n.props);tables.push(n);}
 if(n.type==='EvidenceChart')charts.push(n);
 walk(n.props?.children);
}
walk(module.exports.ReportContent());
assert.equal(tables.length,14);assert.equal(charts.length,2);
const table=id=>tables.find(t=>t.props.id===id).props;
const candidateColumns=Object.fromEntries(table('sac-opt-candidates').children.props.columns.map(c=>[c.field,c]));
const rawParameters={learning_rate:.0001,gamma:.995,robust_score:.1234567},rawParametersBefore=JSON.stringify(rawParameters);
assert.equal(candidateColumns.learning_rate.renderCell(.0001),'1e-4');
assert.equal(candidateColumns.learning_rate.renderCell(.0003),'3e-4');
assert.equal(candidateColumns.gamma.renderCell(.995),'0.995');
assert.equal(candidateColumns.gamma.renderCell(.99),'0.990');
assert.equal(candidateColumns.robust_score.renderCell(.1234567),'0.1235');
assert.equal(candidateColumns.robust_score.renderCell(-.1234567),'-0.1235');
assert.equal(candidateColumns.robust_score.renderCell(.000049),'0.0000');
for(const field of Object.keys(rawParameters)){
 assert.equal(typeof rawParameters[field],'number');
 candidateColumns[field].renderCell(rawParameters[field]);
 assert.equal(candidateColumns[field].renderCell(null),'—');
}
assert.equal(JSON.stringify(rawParameters),rawParametersBefore);
assert.equal(table('sac-opt-candidates').sourceRows,queries.validation_candidates);
assert.equal(table('sac-opt-candidates').displayRows,queries.validation_candidates);
assert.equal(table('sac-opt-main').displayRows.length,4);assert.equal(table('sac-opt-stress').displayRows.length,4);
assert.equal(table('sac-opt-training').displayRows.length,120);
assert.ok(table('sac-opt-training').displayRows.every(r=>r.algorithm==='sac'));
assert.ok(table('sac-opt-gates').sourceRows.every(r=>r.passed===false));
assert.ok(table('sac-opt-gates').displayRows.every(r=>r.passed_label==='原参数参考／不适用'));
assert.equal(table('sac-opt-months').displayRows.length,13);
assert.ok(table('sac-opt-months').displayRows.every(r=>r.before===r.after));
const scatter=charts.find(c=>c.props.id==='sac-opt-risk-return').props;
assert.equal(scatter.rows.length,60);assert.equal(scatter.sourceRows.length,60);
assert.equal(new Set(scatter.rows.map(r=>r['版本'])).size,2);assert.ok(scatter.rows.every(r=>r.symbol==='BTCUSDT'&&r.name.includes('seed')));
const curve=charts.find(c=>c.props.id==='sac-opt-actual-curves').props;
assert.equal(curve.rows.length,366);assert.equal(curve.sourceRows.length,732);
assert.ok(curve.sourceRows.every(r=>r.symbol==='BTCUSDT'&&r.seed===0&&r.scenario==='base'));
assert.ok(curve.rows.every(r=>r['原SAC(%)']===r['验证选参后SAC(%)']));
const broken={...table('sac-opt-training')};delete broken.displayRows;assert.throws(()=>validate(broken),/explicit displayRows/);
assert.equal(JSON.stringify(original),beforeJSON);
console.log(JSON.stringify({passed:true,tables:tables.length,charts:charts.length,actualOriginalRecords:120,explicitDisplayRows:true,noReportWritten:true}));
'''
    completed=subprocess.run(['wsl.exe','-d','Ubuntu-26.04','--exec','/mnt/c/Users/sunny/.cache/codex-research-tools/btceth-rl-20261029/node-v24.19.0-linux-x64/bin/node','--input-type=module','-'],
        input=script,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90,check=True)
    result=json.loads(completed.stdout.strip().splitlines()[-1])
    assert result=={'passed':True,'tables':14,'charts':2,'actualOriginalRecords':120,'explicitDisplayRows':True,'noReportWritten':True}
