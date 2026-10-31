"""Comparison-row lineage and real retained metadata gates; no learner or report build."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import btc_rl_seed_stability_report_app as btc
from scripts import btceth_seed_stability_report_app as joined

ROOT=Path(__file__).resolve().parents[1]
RETAINED=ROOT/'results/backtests/single_pair_rl_six_20261030'


def structural_parts(count=2):
    """Identity-only unit records, with no fabricated financial measurements."""
    parts={}
    for symbol in ('BTCUSDT','ETHUSDT'):
        queries={key:{'rows':[{'run_id':f'identity-fixture:{symbol}:{key}:{i}','value_identity':str(i)} for i in range(count)]} for key in joined.QUERY_COUNTS}
        parts[symbol]=({'queries':queries},{'unit_fixture':True})
    return parts


def test_merge_adds_symbol_without_modifying_source_or_pooling(monkeypatch):
    monkeypatch.setattr(joined,'QUERY_COUNTS',{key:4 for key in joined.QUERY_COUNTS})
    parts=structural_parts();before=deepcopy(parts)
    result=joined.merge_verified_snapshots(parts)
    assert parts==before
    assert set(result)==set(joined.QUERY_COUNTS)
    for query in result.values():
        assert len(query['rows'])==4
        assert [r['symbol'] for r in query['rows']]==['BTCUSDT','BTCUSDT','ETHUSDT','ETHUSDT']
        assert all(r['value_identity'] in ('0','1') for r in query['rows'])


@pytest.mark.parametrize('failure',['missing_coin','foreign_symbol','missing_row','duplicate_account_id'])
def test_join_rejects_wrong_population_or_collisions(monkeypatch,failure):
    monkeypatch.setattr(joined,'QUERY_COUNTS',{key:4 for key in joined.QUERY_COUNTS})
    parts=structural_parts()
    if failure=='missing_coin':del parts['ETHUSDT']
    elif failure=='foreign_symbol':parts['BTCUSDT'][0]['queries']['account_summary']['rows'][0]['symbol']='ETHUSDT'
    elif failure=='missing_row':parts['ETHUSDT'][0]['queries']['account_summary']['rows'].pop()
    else:parts['ETHUSDT'][0]['queries']['account_summary']['rows'][0]['run_id']=parts['BTCUSDT'][0]['queries']['account_summary']['rows'][0]['run_id']
    with pytest.raises(ValueError):joined.merge_verified_snapshots(parts)


def test_completed_eth_reader_and_ui_are_immutable():
    assert joined.sha(ROOT/'scripts/eth_rl_seed_stability_report_app.py')==joined.ETH_READER_SHA256
    assert joined.sha(ROOT/'scripts/eth_rl_seed_stability_report_app.jsx')=='38b8c7383b1c7b84a7a48fb86a6531d76565232889cc18ed620a06b281b76d25'
    assert joined.sha(ROOT/'scripts/eth_rl_seed_stability_report_app.css')=='04d01f5825e2d2d43db159aa35b9e4854b4d4f421aee35cabe9aeadc774b67d9'


def test_btc_expected_accounts_are_only_btc():
    for split in ('validation','test'):
        cases=btc.expected_cases(split)
        assert len(cases)==128
        assert {r[0] for r in cases.values()}=={'BTCUSDT'}
        assert sum(r[2] is None for r in cases.values())==8
        assert sum(r[2] is not None and r[2]>=5 for r in cases.values())==100


def test_actual_btc_retained_ten_counts_and_hash_only_types():
    if not (RETAINED/'fit_complete.json').exists():pytest.skip('Original six-algorithm evidence unavailable')
    spec=btc.read(RETAINED/'protocol.json')
    assert spec['symbols']==['BTCUSDT','ETHUSDT']
    for algorithm in btc.ALGORITHMS:
        for seed in range(5):
            record=btc.core.validate_training_record(RETAINED,'BTCUSDT',algorithm,seed,spec)
            btc.verify_native_training_counters(record,spec)
    zip_path=next((RETAINED/'training/BTCUSDT/sac/seed0/checkpoint').glob('*.zip'))
    names=[zip_path.relative_to(RETAINED).as_posix(),'evaluation/test/BTCUSDT/sac/seed0/base/fills.csv','evaluation/test/BTCUSDT/sac/seed0/base/equity_1h.csv.gz']
    pins={}
    for name in names:
        assert btc.pin_original(RETAINED,name,pins,parse=False) is None
        assert pins[str(RETAINED/name)]==btc.sha(RETAINED/name)


def test_actual_prepared_btc_reaches_missing_new_record_without_report():
    study=ROOT/'results/backtests/btc_rl_seed_stability_20261031'
    if not (study/'training_import.json').exists():pytest.skip('Actual prepared BTC evidence not yet collected')
    missing=next((study/f'training/BTCUSDT/sac/seed{s}/record.json' for s in range(5,30) if not (study/f'training/BTCUSDT/sac/seed{s}/record.json').exists()),None)
    if missing is None:pytest.skip('All new SAC models collected; partial-study rejection no longer applies')
    before=(study/'report').exists()
    with pytest.raises(FileNotFoundError) as result:btc.make_reviewed_snapshot(study,RETAINED)
    assert Path(result.value.filename)==missing
    assert (study/'report').exists()==before


def test_combined_contract_includes_original_dates_costs_and_budgets():
    assert {'train','validation','test','costs','capital_per_account','capital_shared_between_pairs','environment','timesteps_per_model','recipes','dataset_sha256','seeds'}<=set(joined.COMMON_CONTRACT)
    assert joined.QUERY_COUNTS['account_summary']==256
    assert joined.QUERY_COUNTS['training_records']==120
    assert joined.QUERY_COUNTS['cohort_statistics']==16
    assert joined.QUERY_COUNTS['equity_history']==256*366
    assert joined.QUERY_COUNTS['monthly_returns']==256*13


def test_actual_report_training_display_rows_and_multisource_contract():
    """Compile authored JSX with the official compiler, then inspect actual props.

    Public controls are inert sentinels; this is not a browser/render claim.
    The 120 training rows come from the genuine completed first-version report.
    """
    reviewed=ROOT/'results/backtests/btc_rl_seed_stability_20261031/report/reviewed.json'
    if not reviewed.exists() or shutil.which('wsl.exe') is None:
        pytest.skip('Actual completed comparison snapshot and WSL compiler required')
    script=r'''
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {loadPrebuiltCompiler} from 'file:///mnt/c/Users/sunny/.codex/plugins/cache/openai-curated-remote/data-analytics/1.1.1-exp-components.2/output_formats/html/scripts/data-app-runtime.mjs';
const snapshot=JSON.parse(await readFile('/mnt/h/PycharmProjects/hummingbot/results/backtests/btc_rl_seed_stability_20261031/report/reviewed.json','utf8'));
const source=await readFile('/mnt/h/PycharmProjects/hummingbot/scripts/btceth_seed_stability_report_app.jsx','utf8');
const compiler=await loadPrebuiltCompiler();
const compiled=compiler.transform(source,{filePath:'ReportContent.jsx'});
const original=snapshot.queries.training_records.rows,before=JSON.stringify(original);
const jsx=(type,props)=>({type,props:props??{}});
const react={useMemo:f=>f(),useState:value=>[value,()=>{}]};
const publicComponents=Object.fromEntries(['DataComponent','DataTable','Dropdown','EvidenceChart','ReportSection','RichNarrative'].map(n=>[n,n]));
publicComponents.useDataApp=()=>({appTitle:snapshot.title,canEdit:false,mode:'view',reviewedPeriodRows:id=>snapshot.queries[id].rows});
const module={exports:{}};
vm.runInNewContext(compiled,{module,exports:module.exports,require:name=>{
 if(name==='react')return react;
 if(name==='react/jsx-runtime')return {jsx,jsxs:jsx,Fragment:'Fragment'};
 if(name==='../../data-app-public.jsx')return publicComponents;
 if(name.endsWith('.css'))return {};
 throw Error('Unexpected authored import '+name);
}});
const display=module.exports.trainingDisplayRows(original);
assert.equal(display.length,120);assert.equal(new Set(display.map(r=>r.key)).size,120);
for(let i=0;i<original.length;i++){
 const raw=original[i],row=display[i];
 for(const key of Object.keys(raw))assert.equal(JSON.stringify(row[key]),JSON.stringify(raw[key]));
 assert.equal(row.key,`${raw.symbol}:${raw.algorithm}:${raw.seed}`);
 assert.equal(row.algorithm_label,raw.algorithm.toUpperCase());
 assert.equal(row.origin_label,raw.origin==='retained'?'原5个种子':'新增25个种子');
}
const multis=[];
function validate(props){
 assert.ok(Object.hasOwn(props,'displayRows')&&Array.isArray(props.displayRows),'explicit displayRows required');
 assert.deepEqual(Object.keys(props.sourceRowsByQuery).sort(),[...props.queryIds].sort());
 const table=props.children;assert.equal(table.type,'DataTable');assert.equal(table.props.rows,props.displayRows);
}
function walk(node){
 if(Array.isArray(node)){node.forEach(walk);return;}
 if(!node||typeof node!=='object')return;
 if(typeof node.type==='function'){walk(node.type(node.props));return;}
 if(node.type==='DataComponent'&&node.props.queryIds?.length>1){validate(node.props);multis.push(node);}
 walk(node.props?.children);
}
walk(module.exports.ReportContent());
assert.equal(multis.length,1);assert.equal(multis[0].props.id,'seed-training');
assert.equal(multis[0].props.displayRows.length,120);
assert.equal(multis[0].props.sourceRowsByQuery.training_records,original);
assert.equal(multis[0].props.sourceRowsByQuery.training_import,snapshot.queries.training_import.rows);
const broken={...multis[0].props};delete broken.displayRows;
assert.throws(()=>validate(broken),/explicit displayRows required/);
assert.equal(JSON.stringify(original),before);
console.log(JSON.stringify({passed:true,actualTrainingRows:120,multiQueryComponents:1,displayRowsSharedWithTable:true,allOriginalFieldsUnchanged:true,originalMissingPropRejected:true,noReportWritten:true}));
'''
    completed=subprocess.run(['wsl.exe','-d','Ubuntu-26.04','--exec','/mnt/c/Users/sunny/.cache/codex-research-tools/btceth-rl-20261029/node-v24.19.0-linux-x64/bin/node','--input-type=module','-'],
        input=script,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90,check=True)
    result=json.loads(completed.stdout.strip().splitlines()[-1])
    assert result['passed'] is True and result['actualTrainingRows']==120
    assert result['multiQueryComponents']==1 and result['displayRowsSharedWithTable'] is True
    assert result['allOriginalFieldsUnchanged'] is True and result['originalMissingPropRejected'] is True
