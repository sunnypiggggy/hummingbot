"""Arithmetic/path/readback gates only; artificial unit fixtures never become a report."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts import eth_rl_seed_stability_report_app as report


def arithmetic_rows():
    rows=[]
    for algorithm in report.ALGORITHMS:
        for seed in report.SEEDS:
            for scenario in report.core.COSTS:
                rows.append({"algorithm":algorithm,"seed":seed,"scenario":scenario,
                    "net_return":(seed-15)/100,"max_drawdown":-seed/100,
                    "run_id":f"arithmetic-fixture-{algorithm}-{seed}-{scenario}"})
    return rows


def test_linear_quantiles_and_full_new_denominators():
    stats=report.cohort_statistics(arithmetic_rows())
    assert len(stats)==8
    new=next(r for r in stats if r['cohort']=='new25' and r['algorithm']=='sac' and r['scenario']=='base')
    all_=next(r for r in stats if r['cohort']=='all30' and r['algorithm']=='sac' and r['scenario']=='base')
    assert (new['seeds'],new['positive_seeds'],new['loss_seeds'],new['zero_seeds'])==(25,14,10,1)
    assert (all_['seeds'],all_['positive_seeds'],all_['loss_seeds'],all_['zero_seeds'])==(30,14,15,1)
    assert [new[k] for k in ('min_return','p10_return','q1_return','median_return','q3_return','p90_return','max_return')]==pytest.approx([-.10,-.076,-.04,.02,.08,.116,.14])
    assert [all_[k] for k in ('p10_return','q1_return','median_return','q3_return','p90_return')]==pytest.approx([-.121,-.0775,-.005,.0675,.111])
    assert (new['median_drawdown'],new['p90_drawdown'],new['worst_drawdown'])==pytest.approx((.17,.266,.29))
    assert (all_['median_drawdown'],all_['p90_drawdown'],all_['worst_drawdown'])==pytest.approx((.145,.261,.29))
    assert new['drawdown_above_20_seeds']==9


@pytest.mark.parametrize('mutation',['missing','duplicate','foreign_seed','nan_return','infinite_drawdown'])
def test_incomplete_or_nonfinite_distribution_rejected(mutation):
    rows=arithmetic_rows()
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows.append(deepcopy(rows[-1]))
    elif mutation=='foreign_seed':rows[-1]['seed']=30
    elif mutation=='nan_return':rows[-1]['net_return']=float('nan')
    else:rows[-1]['max_drawdown']=float('inf')
    with pytest.raises(ValueError):report.cohort_statistics(rows)


def test_same_label_differences_do_not_average_accounts():
    rows=arithmetic_rows()
    for row in rows:
        if row['algorithm']=='td3':row['net_return']+=.01;row['max_drawdown']-=.02
    pairs=report.paired_seed_results(rows);stats=report.paired_statistics(pairs)
    assert len(pairs)==60 and len(stats)==4
    assert all(r['return_difference']==pytest.approx(.01) and r['drawdown_difference']==pytest.approx(.02) for r in pairs)
    assert all(r['median_return_difference']==pytest.approx(.01) and r['median_drawdown_difference']==pytest.approx(.02) for r in stats)
    assert sorted(r['seeds'] for r in stats)==[25,25,30,30]
    assert all(r['td3_higher_return_seeds']==r['seeds'] and r['td3_lower_drawdown_seeds']==r['td3_both_better_seeds']==0 for r in stats)


@pytest.mark.parametrize('bad',['../outside.json','/absolute.json','C:/drive.json','a\\b.json','a/./b.json',''])
def test_evidence_paths_require_canonical_contained_name(tmp_path,bad):
    with pytest.raises(ValueError):report.relative(tmp_path,bad)


def test_exact_expected_account_universe():
    for split in ('validation','test'):
        cases=report.expected_cases(split)
        assert len(cases)==128
        assert sum(r[2] is None for r in cases.values())==8
        assert sum(r[2] is not None and r[2]>=5 for r in cases.values())==100
        assert all(r[0]=='ETHUSDT' for r in cases.values())
        assert all(r[1] in ('sac','td3') or r[2] is None for r in cases.values())


def test_original_and_new_study_cannot_be_same(tmp_path):
    with pytest.raises(ValueError,match='Separate'):
        report.make_reviewed_snapshot(tmp_path,tmp_path)


def test_actual_retained_ten_native_counts_and_episode_boundaries():
    source=Path(__file__).resolve().parents[1]/'results/backtests/single_pair_rl_six_20261030'
    if not (source/'fit_complete.json').exists():pytest.skip('Actual retained research evidence unavailable')
    spec=report.read(source/'protocol.json')
    checked=0
    for algorithm in report.ALGORITHMS:
        for seed in range(5):
            record=report.core.validate_training_record(source,'ETHUSDT',algorithm,seed,spec)
            report.verify_native_training_counters(record,spec)
            checked+=1
    assert checked==10


def test_actual_original_zip_csv_gz_are_hash_only():
    source=Path(__file__).resolve().parents[1]/'results/backtests/single_pair_rl_six_20261030'
    if not (source/'fit_complete.json').exists():pytest.skip('Actual retained research evidence unavailable')
    zip_file=next((source/'training/ETHUSDT/sac/seed0/checkpoint').glob('*.zip'))
    names=[zip_file.relative_to(source).as_posix(),
        'evaluation/test/ETHUSDT/sac/seed0/base/fills.csv',
        'evaluation/test/ETHUSDT/sac/seed0/base/equity_1h.csv.gz']
    pins={}
    for name in names:
        assert report.pin_original(source,name,pins,parse=False) is None
        assert pins[str(source/name)]==report.sha(source/name)
    assert report.pin_original(source,'protocol.json',pins)['training_model_count']==60


def test_actual_prepared_import_reaches_missing_fresh_record_without_report():
    root=Path(__file__).resolve().parents[1]
    study=root/'results/backtests/eth_rl_seed_stability_20261031'
    source=root/'results/backtests/single_pair_rl_six_20261030'
    if not (study/'training_import.json').exists():pytest.skip('Actual prepared seed-extension evidence unavailable')
    missing=next((study/f'training/ETHUSDT/sac/seed{s}/record.json' for s in range(5,30)
        if not (study/f'training/ETHUSDT/sac/seed{s}/record.json').exists()),None)
    if missing is None:pytest.skip('Fresh SAC models already collected; partial-study rejection no longer applicable')
    existed=(study/'report').exists()
    with pytest.raises(FileNotFoundError) as failure:
        report.make_reviewed_snapshot(study,source)
    assert Path(failure.value.filename)==missing
    assert (study/'report').exists()==existed
