"""Read-only report adapter for the explicitly parallel DroQ evidence.

The legacy reader's financial/account readbacks are reused unchanged. New
process metadata is checked as itself: an honestly failed original queue, a
completed accelerated cohort, 24 case workers and a sealed two-GPU annual
dispatch. No process proof is renamed or synthesized for legacy validation.
"""
from __future__ import annotations
import copy
from pathlib import Path

import pandas as pd

from scripts import btceth_rl_report_droq as h
from scripts.collect_btceth_rl_droq_parallel import required_files

REMOTE_AUDIT = 'independent_parallel_audit.json'
LOCAL_AUDIT = 'independent_parallel_audit.local.json'
VALIDATION_AUDIT = 'independent_parallel_audit.validation.json'
SCHEDULER_SHA = '52ba244d55abf36fc49cdbaa8374c179c939dc35b07f3d27bbd8c7a07b233453'
PORTABILITY_SCOPE = 'stored POSIX performance path only'


def validation_files():
    return (h.validation_files() - {'validate_process.json', 'validate.log'}) | {
        'parallel_validation_process.json', 'parallel_validation_runner.log',
        'parallel_validation_authorization.json', 'parallel_validation_resource_preflight.json',
        'resource_batch_process.json'}


def external_pins(sources, audit, mapping):
    absolute = audit.get('external_input_sha256') or {}
    relative = audit.get('extension_source_sha256') or {}
    pins = {**absolute, **relative}
    if not pins:
        raise ValueError('Parallel audit must bind its explicitly changed execution sources')
    for name, digest in pins.items():
        if name in relative and name not in mapping:
            raise ValueError('Every relative extension label needs an explicit local path: ' + name)
        p = Path(mapping[name]) if name in mapping else Path(name)
        if not p.is_absolute() or not p.is_file() or h.sha(sources.file(p)) != digest:
            raise ValueError('External audit evidence needs an exact local mapping: ' + name)


def local_portability_source(sources, audit, source_root, *, local):
    """Keep a narrowly declared local metadata adapter in report provenance."""
    value = audit.get('local_metadata_portability')
    if value is None:
        return
    if not local or not isinstance(value, dict) or value.get('scope') != PORTABILITY_SCOPE:
        raise ValueError('Only the explicit local stored-POSIX-path portability scope is supported')
    name = value.get('adapter_file')
    if not isinstance(name, str) or not Path(name).is_absolute():
        raise ValueError('Local metadata adapter needs an explicit absolute source path')
    path = h.no_link(name).resolve(); source_root = h.no_link(source_root).resolve()
    allowed = [source_root / 'scripts', source_root / 'results/backtests/btceth_rl_droq_20261030_preflight/performance']
    if not any(path.is_relative_to(h.no_link(root).resolve()) for root in allowed):
        raise ValueError('Local metadata adapter must be in the provided source scripts or owned preflight performance directory')
    if h.sha(sources.file(path)) != value.get('adapter_sha256'):
        raise ValueError('Local metadata adapter source SHA differs from its audit declaration')


def execution_sources(sources, selection, mapping):
    pins = selection.get('extension_source_sha256')
    if not isinstance(pins, dict) or not pins:
        raise ValueError('Sealed parallel selector must bind its new execution sources')
    # Selection stores source labels. Actual local locations come exclusively
    # from the explicit remote-to-local audit map, and must be unambiguous.
    values = set()
    for name, digest in pins.items():
        if name not in mapping or h.sha(mapping[name]) != digest:
            raise ValueError('Sealed extension needs one explicit external source: ' + name)
        sources.file(mapping[name]); values.add(digest)
    return pins, values


def process_metadata(sources, study, p, selection, mapping):
    old = sources.json(study / 'fit_process.json')
    accelerated = sources.json(study / 'accelerated_fit_process.json')
    scheduling = sources.json(study / 'scheduling_audit.fit.json')
    external_pins(sources, scheduling, mapping)
    complete = sources.json(study / 'accelerated_fit_completion.json')
    if (old.get('status') != 'failed' or type(old.get('exit_code')) is not int or old['exit_code'] <= 0
            or old.get('log_sha256') != h.sha(sources.file(study / 'fit.log'))
            or accelerated.get('status') != 'completed' or type(accelerated.get('exit_code')) is not int or accelerated['exit_code'] != 0
            or accelerated.get('worker_exit_code') != 0 or accelerated.get('source_sha256') != SCHEDULER_SHA
            or accelerated.get('log_sha256') != h.sha(sources.file(study / 'accelerated_seed4.log'))
            or accelerated.get('completion_sha256') != h.sha(study / 'accelerated_fit_completion.json')
            or complete.get('training_completion_sha256') != h.sha(study / 'training_completion.json')
            or scheduling.get('passed') is not True or scheduling.get('status') != 'passed' or scheduling.get('training_model_count') != 5
            or scheduling.get('original_collision_verified') is not True or scheduling.get('original_0_1_preserved') is not True
            or scheduling.get('original_2_3_start_instances_preserved') is not True):
        raise ValueError('Actual original collision and independently verified accelerated completion required')
    span = scheduling.get('derived_cohort_envelope', {})
    if (span.get('kind') != 'derived_training_time_bounds_only' or span.get('original_terminal_relabelled') is not False
            or span.get('original_terminal_status') != 'failed' or span.get('accelerated_terminal_status') != 'completed'
            or span.get('started_at') != old.get('started_at') or span.get('completed_at') != accelerated.get('completed_at')
            or span.get('original_fit_process_sha256') != h.sha(study / 'fit_process.json')
            or span.get('accelerated_process_sha256') != h.sha(study / 'accelerated_fit_process.json')):
        raise ValueError('Derived cohort time bounds must retain both real process identities')
    for phase in (accelerated,):
        if (not h.positive_int(phase.get('supervisor_pid')) or not h.positive_int(phase.get('child_pid'))
                or phase['supervisor_pid'] == phase['child_pid']):
            raise ValueError('Actual distinct accelerated process identities required')
    validation = sources.json(study / 'parallel_validation_process.json')
    batch = sources.json(study / 'resource_batch_process.json')
    vcomplete = sources.json(study / 'validation_completion.json')
    if (validation.get('status') != 'completed' or validation.get('completed_cases') != validation.get('total_cases') or validation.get('total_cases') != 24
            or validation.get('completion_sha256') != h.sha(study / 'validation_completion.json')
            or validation.get('original_runner_sha256') != h.RUNNER_SHA256
            or batch.get('status') != 'completed_validation_awaiting_financial_audit' or batch.get('exit_code') != 0
            or batch.get('child_exit_code') != 0 or batch.get('child_pid') != validation.get('coordinator_pid')
            or batch.get('pid') != validation.get('parent_pid') or not h.positive_int(validation.get('coordinator_pid'))
            or batch.get('log_sha256') != h.sha(sources.file(study / 'parallel_validation_runner.log'))
            or vcomplete.get('case_count') != vcomplete.get('repeat_count') or vcomplete.get('case_count') != 24):
        raise ValueError('Actual24-worker validation completion and supervisor required')
    test = sources.json(study / 'parallel_test_process.json')
    tcomplete = sources.json(study / 'test_completion.json')
    extension, extension_values = execution_sources(sources, selection, mapping)
    if (test.get('status') != 'completed' or type(test.get('exit_code')) is not int or test['exit_code'] != 0
            or not h.positive_int(test.get('supervisor_pid')) or not h.positive_int(test.get('child_pid'))
            or test['supervisor_pid'] == test['child_pid'] or test.get('source_sha256') not in extension_values
            or test.get('log_sha256') != h.sha(sources.file(study / 'parallel_test_runner.log'))
            or tcomplete.get('case_count') != tcomplete.get('repeat_count') or tcomplete.get('case_count') != 12):
        raise ValueError('Real source-bound parallel annual terminal0 and twelve repeats required')
    annex = sources.json(study / 'parallel_annual_resource_annex.json')
    assignments = annex.get('case_gpu_assignments', {})
    expected = {f"test__{cost}__{selection['candidate']['family']}__droq__balanced__{key}"
                for cost in h.COSTS for key in (*map(str, h.SEEDS), 'ensemble')}
    uuids = {g['uuid'] for g in p['physical_gpus']}
    if set(assignments) != expected or set(assignments.values()) != uuids or any(list(assignments.values()).count(u) != 6 for u in uuids):
        raise ValueError('The twelve original accounts must be preassigned six to each original V100')
    family = selection['candidate']['family']; group = sources.json(study / f'test/{family}/balanced/completion.json')
    cases = group.get('case_workers', []); seen = set()
    if group.get('device') is not None or len(cases) != 12:
        raise ValueError('Two-card annual group cannot impersonate a single worker')
    for worker in cases:
        run_id = worker.get('run_id')
        if run_id in seen or run_id not in assignments or not h.positive_int(worker.get('worker_pid')) or worker.get('device', {}).get('visible_gpu_uuid') != assignments[run_id]:
            raise ValueError('Annual case PID/card differs from its resource annex')
        seen.add(run_id)
    return span, validation, test, tcomplete


def load_droq(study_root, main_root, *, audit_path=None, source_root=h.ROOT, external_map=None):
    study, main, source_root = map(lambda x: h.no_link(x).resolve(), (study_root, main_root, source_root))
    if study == main or study.is_relative_to(main) or main.is_relative_to(study):
        raise ValueError('Parallel DroQ preserves the original study in an independent tree')
    sources = h.Sources(); roots = dict(study=study, main=main, source=source_root)
    if external_map is None:
        receipt = sources.json(study / 'parallel_transfer_receipt.json')
        external_map = receipt.get('external_path_map', {})
    paths = [h.no_link(audit_path).resolve() if audit_path else study / LOCAL_AUDIT, study / REMOTE_AUDIT]
    paths[0].relative_to(study)
    if paths[0] == paths[1] or h.sha(paths[0]) == h.sha(paths[1]):
        raise ValueError('Independent local and retained remote full26 audits must remain distinct')
    selection, seal = (sources.json(study / n) for n in ('selection.json', 'selection_receipt.json'))
    chosen = selection['candidate']; family = chosen['family']; audits = []
    for i, path in enumerate(paths):
        audit = sources.json(path); h._audit(audit)
        local_portability_source(sources, audit, source_root, local=i == 0)
        h._bind(sources, roots, audit, study_required=required_files(family), main_required={'frozen_spec.json', 'runtime.json', 'independent_audit.json'} | h.baseline_files())
        external_pins(sources, audit, external_map); audits.append(audit)
    p, _ = h._protocol(sources, study, main, source_root)
    if any(a.get('dataset_sha256') != p['dataset_sha256'] or a.get('source_sha256') != p['source_sha256'] for a in audits):
        raise ValueError('Both full financial audits must describe the unchanged dataset and9 original sources')
    fit, validate, test, completion = process_metadata(sources, study, p, selection, external_map)
    initial = sources.json(study / VALIDATION_AUDIT); h._audit(initial, validation=True)
    h._bind(sources, roots, initial, study_required=validation_files() | h.training_files(), main_required={'frozen_spec.json', 'runtime.json'})
    external_pins(sources, initial, external_map)
    training = h._training(sources, study, p, fit, [*audits, initial])
    if (seal.get('selection_file') != 'selection.json' or seal.get('selection_sha256') != h.sha(study / 'selection.json')
            or seal.get('protocol_sha256') != selection.get('protocol_sha256') or selection.get('protocol_sha256') != h.sha(study / 'protocol.json')
            or selection.get('audit_file') != VALIDATION_AUDIT or selection.get('audit_sha256') != h.sha(study / VALIDATION_AUDIT)
            or selection.get('seed_selection') is not False or selection.get('selection_uses_test') is not False
            or selection.get('test_results_available_to_selector') is not False or any(selection.get(k) is not True for k in h.FLAGS)):
        raise ValueError('Validation-only original two-budget choice must be sealed before the annual run')
    sources.pins(study, selection.get('training_source_sha256'), required=h.training_files() - {n for n in h.training_files() if n.endswith('/started.json')})
    sources.pins(study, selection.get('validation_source_sha256'), required=validation_files())
    actual, frames = {}, []
    for f in h.BUDGETS:
        frame = sources.csv(study / f'validation/{f}/balanced/summary.csv'); actual[(f, 'balanced')] = h._candidate(frame, f, 'balanced'); frames.append(frame)
    h._runs(initial, 'runs', pd.concat(frames, ignore_index=True), hours=7080)
    pool = selection.get('candidates', [])
    if len(pool) != 2 or {(x.get('family'), x.get('profile')) for x in pool} != set(actual):
        raise ValueError('Exactly two predeclared validation candidates required')
    for item in pool: h.same_metrics(item, actual[(item['family'], item['profile'])], 'Original fixed validation score')
    if pool != sorted(pool, key=lambda x: (-x['score'], 0 if x['family'] == 'raw' else 1, x['profile'])) or chosen != pool[0]:
        raise ValueError('The original score and raw-first tie order must remain unchanged')
    auth, marker = (sources.json(study / n) for n in ('annual_authorization.json', 'test_started.json'))
    if (auth.get('annual_evaluation_authorized') is not True or auth.get('financial_parameters_fixed') is not True or auth.get('no_live_trading') is not True
            or auth.get('selection_sha256') != h.sha(study / 'selection.json') or any(auth.get(k) is not True for k in h.FLAGS)
            or marker.get('protocol_sha256') != h.sha(study / 'protocol.json')
            or marker.get('authorization_sha256') != h.sha(study / 'annual_authorization.json') or marker.get('selection_sha256') != h.sha(study / 'selection.json')
            or completion.get('candidate') != chosen or completion.get('selection_sha256') != h.sha(study / 'selection.json')
            or not (h.utc(fit['completed_at']) <= h.utc(validate['started_at']) <= h.utc(validate['completed_at']) <= h.utc(initial['audited_at'])
                <= h.utc(selection['selected_at']) <= h.utc(seal['sealed_at']) <= h.utc(auth['authorized_at']) <= h.utc(marker['at']) <= h.utc(test['started_at']))):
        raise ValueError('Actual complete validation and sealing must precede separately authorized annual execution')
    main_audit = sources.json(main / 'independent_audit.json')
    if (main_audit.get('passed') is not True or main_audit.get('status') != 'passed' or main_audit.get('test_complete') is not True
            or main_audit.get('run_count') != 172 or not isinstance(main_audit.get('runs'), dict) or len(main_audit['runs']) != 172
            or not h.positive_int(main_audit.get('checks_total')) or main_audit.get('failure_count') != 0 or main_audit.get('failures') != []):
        raise ValueError('Original172 reference audit must remain complete')
    sources.pins(main, main_audit.get('input_sha256'), required={'frozen_spec.json', 'test_completion.json'} | h.baseline_files())
    annual = [sources.csv(study / f'test/{family}/balanced/{n}') for n in h.GROUP_FILES[:3]]
    baseline = [sources.csv(main / f'evaluation/test/baselines/{n}') for n in h.GROUP_FILES[:3]]
    h._summary(annual[0], split='test', family=family, profile='balanced'); h._summary(baseline[0], split='test', baseline=True)
    for group in (annual, baseline): h._history(group[1], group[2], group[0])
    targets = h._targets(annual[0], baseline[0], chosen)
    for audit in audits:
        h._runs(audit, 'runs', annual[0], hours=8760); refs = h._runs(audit, 'baseline_runs', baseline[0], hours=8760)
        if h.utc(audit['audited_at']) < h.utc(test['completed_at']) or any(v['fingerprints'] != main_audit['runs'][k]['fingerprints'] for k, v in refs.items()):
            raise ValueError('Full audit must follow actual completion and preserve original reference fingerprints')
        h.same_metrics(audit.get('research_targets'), targets, 'Own recomputed original five gates')
    h._case_readbacks(sources, study, annual[0], annual[1], audits[0]['runs'], family=family, training=training)
    h._case_readbacks(sources, main, baseline[0], baseline[1], audits[0]['baseline_runs'], baseline=True)
    goal = sources.json(study / 'goal_candidate_evidence.json')
    if (goal.get('audit_sha256') != h.sha(study / REMOTE_AUDIT) or goal.get('accounting_audit_passed') is not True
            or goal.get('protocol_sha256') != h.sha(study / 'protocol.json') or goal.get('selection_sha256') != h.sha(study / 'selection.json')
            or h.utc(goal['at']) < h.utc(audits[1]['audited_at'])):
        raise ValueError('Own goal evidence must bind the genuine remote full26 financial audit')
    for key, value in targets.items(): h.same_metrics(goal.get(key), value, 'Own DroQ goal:' + key)
    return build_queries(sources, paths, audits, p, selection, targets, training, annual, baseline)


def build_queries(sources, paths, audits, p, selection, targets, training, annual, baseline):
    sources.file(Path(__file__)); tables = [[], [], []]
    for role, group in (('droq_selected', annual), ('original_baseline_reference', baseline)):
        for frame, dest in zip(group, tables):
            item = frame.copy(); item['scope'] = h.SCOPE; item['droq_role'] = role; item['training_budget_multiple'] = 1 if role == 'droq_selected' else None; dest.append(item)
    target_rows = [dict(gate=k, met=v, category='已观察年度的DroQ自身诊断') for k, v in targets['gates'].items()]
    target_rows.extend(dict(gate=k, met=v, category='简单趋势对照') for k, v in targets['beats_simple_trend_controls'].items())
    frames = dict(droq_summary=pd.concat(tables[0], ignore_index=True), droq_equity=pd.concat(tables[1], ignore_index=True), droq_monthly=pd.concat(tables[2], ignore_index=True),
        droq_targets=pd.DataFrame(target_rows), droq_validation_selection=pd.DataFrame(selection['candidates']), droq_training=pd.DataFrame(training),
        droq_audit=pd.DataFrame([dict(role='本机明确审计' if i == 0 else '保留远端审计', file=str(path), sha256=h.sha(path), passed=True,
            new_runs=12, baseline_reference_runs=14, validation_runs=24, training_models=5, repeat_count=12, fresh_holdout_success=False) for i, path in enumerate(paths)]))
    source = dict(label='DroQ原作者数学、明确并行调度与已观察年度研究', classification='modeled', timezone='UTC', period=dict(start=h.TEST[0], endExclusive=h.TEST[1]),
        files=list(sources.hashes), fileHashes=dict(sources.hashes), datasetSha256=p['dataset_sha256'], caveats=[
            '原训练队列的冲突失败如实保留，五模型完成由独立加速调度证明；未伪造旧fit/validate终态。',
            '32768环境步×5；G20每模型555360次联合Q更新，不能称与原SAC计算量相同。.01 dropout为预声明研究参数。',
            '新年度已观察，不是fresh holdout；保留原五项目标、五种子和全部成本场景。',
            '验证24账户与24重复并行；年度12账户透明分配两张V100各6，账户与金融公式不变。',
            '年度26行含12新DroQ账户和14原基准引用；不是26个新模型或新增资金。',
            '小时权益决定回撤；日边界曲线、月度和逐笔费用均读回对账。'])
    queries = {}
    for name, frame in frames.items():
        context = copy.deepcopy(source)
        if name in ('droq_training', 'droq_validation_selection'):
            dates = h.TRAIN if name == 'droq_training' else h.VALIDATION; context['period'] = dict(start=dates[0], endExclusive=dates[1])
        queries[name] = dict(rows=frame.astype(object).where(pd.notna(frame), None).to_dict('records'), source=context,
            methods=[dict(language='text', code='Read-only exact SHA-bound parallel DroQ evidence, full26 financial accounts,24 validation accounts and5 models. Original financial CSV readbacks and five gates reused unchanged; real parallel process evidence is explicit.')])
    sources.unchanged()
    return dict(queries=queries, source=source, targets=targets, selection=selection, audit=audits[0], external_input_sha256=dict(sources.hashes),
        query_run_count=26, new_run_count=12, baseline_reference_run_count=14, validation_run_count=24, training_model_count=5,
        original_scopes_unchanged=True, fresh_holdout_success=False, training_budget_multiple=1, environment_step_budget_multiple=1, compute_budget_comparable=False,
        droqFollowup=dict(algorithm='droq', profile='balanced', scope=h.SCOPE, chosen=selection['candidate'], targets=targets, known_year=True,
            fresh_holdout_success=False, prior_scopes_unchanged=True, models=5, timesteps_per_model=32768, environment_step_budget_multiple=1,
            compute_budget_comparable=False, q_inner_rounds_per_model=555360, individual_q_optimizer_steps_per_model=1110720,
            actor_optimizer_steps_per_model=27768, alpha_optimizer_steps_per_model=27768, upstream_commit=h.UPSTREAM_COMMIT, recipe_is_pure_official_default=False))
