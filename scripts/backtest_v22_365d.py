"""365-day offline v22 replay, preserving original weekly models and provenance."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import xgboost_long_risk_gate_v22 as v22
from grid_v22_recovery_ablation import CounterState, step
from prepare_v22_365d_inputs import START, END, validate, sha

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'results/backtests/v22_365d_20261017'
STRUCTURE = ['roc_48h_4h', 'sqzmom_pct_4h', 'di_spread', 'ema20_slope_atr_12h', 'below_ema20_ratio_72h']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding='utf-8')


def save_csv(path, frame):
    frame.to_csv(path, index=False, float_format='%.17g', lineterminator='\n',
                 compression={'method':'gzip','mtime':0} if path.suffix=='.gz' else None)


def load_models(inputs):
    lock = json.loads((inputs/'production_shadow_lock.json').read_text())
    if sha(inputs/'model.joblib') != lock['model_sha256']:
        raise ValueError('copied production model SHA mismatch')
    with tarfile.open(inputs/'portable_models.tar.gz') as archive:
        bundle = json.load(archive.extractfile('portable_manifest.json'))
        if bundle['source_joblib_sha256'] != lock['model_sha256']:
            raise ValueError('native model export not bound to production model')
        for pair, item in bundle['pairs'].items():
            for week in item['weeks']:
                raw = archive.extractfile(week['ubj_file']).read()
                if hashlib.sha256(raw).hexdigest() != week['model_sha256']:
                    raise ValueError('original weekly model hash mismatch')
                model = XGBClassifier()
                model.load_model(bytearray(raw))
                probe = float(model.predict_proba(np.zeros((1,len(item['features'])),dtype=np.float32), validate_features=False)[0,1])
                if abs(probe-week['zero_probe_probability']) > 1e-7:
                    raise ValueError('cross-platform native prediction parity failed')
                week['model'] = model
                week['provenance'] = 'ORIGINAL_SIGNED_WEEK'
                if week['last_label_ready_ts'] > week['train_cutoff'] or week['development_last_ts'] >= week['calibration_first_ts']:
                    raise ValueError('original weekly label/calibration leakage')
    v22.validate_weekly_bundle(bundle)
    return bundle, lock


def prepare_models(output):
    inputs = output/'inputs'
    bundle, lock = load_models(inputs)
    frames = {pair:pd.read_csv(inputs/f'binance_{pair}_5m.csv') for pair in v22.PAIRS}
    for pair, frame in frames.items():
        validate(frame, int(frame.timestamp.iloc[0]), END)
    original_start = int(lock['effective_start'])
    boundaries = []
    cursor = original_start-7*86400
    # Add one whole warmup week before the requested observation window.
    while cursor >= START-14*86400:
        boundaries.append(cursor)
        cursor -= 7*86400
    boundaries.sort()
    if not boundaries or boundaries[0] > START-7*86400:
        raise ValueError('insufficient state warmup')
    backfill_path = output/'backfill_model_manifest.json'
    training_input_hash = hashlib.sha256((sha(inputs/'market_data_manifest.json')+lock['model_sha256']).encode()).hexdigest()
    if backfill_path.exists():
        cached = json.loads(backfill_path.read_text())
        if cached['training_input_hash'] != training_input_hash:
            raise ValueError('training inputs changed; use a new experiment output directory')
        for pair, item in bundle['pairs'].items():
            weeks = []
            for week in cached['pairs'][pair]:
                model_path = output/week['offline_model_file']
                if sha(model_path) != week['model_sha256']:
                    raise ValueError('cached backfill model changed')
                model = XGBClassifier()
                model.load_model(model_path)
                weeks.append({**week,'model':model})
            item['weeks'] = weeks + item['weeks']
        print('Reused validated offline backfill models', flush=True)
    else:
        from compare_independent_gate_ml_stops import build_feature_panel
        import optimize_xgboost_grid_risk_gate_v7 as labels
        import retrain_xgboost_long_risk_gate_250d_v19 as v19
        import tune_xgboost_momentum_stop_v2 as tune
        tune.XGB_N_JOBS = 2
        training_candles = {p:f[f.timestamp < original_start].copy() for p,f in frames.items()}
        print('Building disjoint training/calibration features', flush=True)
        panel = v19.prepare_panel(labels.relabel_panel(build_feature_panel(training_candles, horizon_hours=6), training_candles))
        saved = {'training_input_hash':training_input_hash,'pairs':{},'offline_only':True,'deployment_allowed':False}
        for pair, item in bundle['pairs'].items():
            frame = v19.target_frame(panel, item['target'], pair)
            new = []
            for index, cutoff in enumerate(boundaries):
                started = time.time()
                model, calibration, audit = v19.fit_leakage_safe(frame, cutoff, item['config'], item['features'])
                threshold = float(calibration.probability.quantile(v22.GATES[pair].entry_quantile))
                fold = index-len(boundaries)+1
                relative = f'offline_models/{pair}/{fold}.ubj'
                target = output/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = bytes(model.get_booster().save_raw(raw_format='ubj'))
                target.write_bytes(raw)
                week = {'fold':fold,'train_cutoff':cutoff,'test_start':cutoff,'test_end':cutoff+7*86400,
                        'entry_threshold':threshold,'calibration_threshold':threshold,
                        'model_sha256':hashlib.sha256(raw).hexdigest(),'provenance':'OFFLINE_EARLY_WEEK_BACKFILL',
                        'offline_model_file':relative, **audit}
                if week['last_label_ready_ts'] > cutoff or week['development_last_ts'] >= week['calibration_first_ts']:
                    raise AssertionError('backfill temporal leakage')
                new.append({**week,'model':model})
                print(pair, 'backfill', pd.Timestamp(cutoff,unit='s',tz='UTC'), 'trees',audit['best_tree_count'],
                      'seconds',round(time.time()-started,1), flush=True)
            saved['pairs'][pair] = [{k:v for k,v in w.items() if k!='model'} for w in new]
            item['weeks'] = new + item['weeks']
        write_json(backfill_path, saved)
    v22.validate_weekly_bundle(bundle)
    windows = [[(w['test_start'],w['test_end']) for w in bundle['pairs'][p]['weeks']] for p in v22.PAIRS]
    if windows[0] != windows[1] or windows[0][0][0] > START or windows[0][-1][1] < END:
        raise ValueError('extended model coverage mismatch')
    return bundle, frames, lock, boundaries[0]


def predict_and_replay(output, bundle, candles, first_signal):
    print('Building causal inference features and hourly predictions', flush=True)
    panel = v22.build_inference_panel(candles)
    predictions = []
    audits = []
    for pair, item in bundle['pairs'].items():
        frame = panel[(panel.pair==pair)&(panel.signal_ts>=first_signal)&(panel.signal_ts<END)].copy().reset_index(drop=True)
        if not np.array_equal(frame.signal_ts.to_numpy(), np.arange(first_signal,END,3600)):
            raise ValueError(f'{pair} hourly feature coverage incomplete')
        frame['probability'] = np.nan
        frame['threshold'] = np.nan
        for week in item['weeks']:
            mask = frame.signal_ts.between(week['test_start'],week['test_end'],inclusive='left')
            if mask.any():
                frame.loc[mask,'probability'] = week['model'].predict_proba(frame.loc[mask,item['features']])[:,1]
                frame.loc[mask,'threshold'] = week['entry_threshold']
                frame.loc[mask,'fold'] = week['fold']
                frame.loc[mask,'week_start'] = week['test_start']
                frame.loc[mask,'provenance'] = week['provenance']
            audits.append({'pair':pair,**{k:v for k,v in week.items() if k!='model'}})
        if frame[['probability','threshold']].isna().any().any():
            raise ValueError('missing model prediction; fallback forbidden')
        predictions.append(frame[['pair','signal_ts','last_complete_4h_ts',*STRUCTURE,'probability','threshold','fold','week_start','provenance']])
    shared = pd.concat(predictions,ignore_index=True)
    save_csv(output/'hourly_predictions.csv.gz',shared)
    rows = []
    for pair in v22.PAIRS:
        state = CounterState()
        for row in shared[shared.pair==pair].itertuples(index=False):
            result = step(state,arm='A',pair=pair,probability=float(row.probability),threshold=float(row.threshold),
                          ts=int(row.signal_ts),structure_ts=int(row.last_complete_4h_ts),
                          structure=tuple(float(getattr(row,k)) for k in STRUCTURE),fold=int(row.fold),boundary=int(row.week_start))
            result.update(provenance=row.provenance,entry_threshold=row.threshold,
                          risk_off_active=result['risk_off'],recommended_buy_enabled=not result['risk_off'])
            rows.append(result)
    signals = pd.DataFrame(rows)
    save_csv(output/'hourly_signals.csv.gz',signals)
    write_json(output/'weekly_model_audit.json',audits)
    return signals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['signals','replay','all'],default='all')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage != 'replay':
        bundle, candles, lock, first = prepare_models(args.output)
        signals = predict_and_replay(args.output,bundle,candles,first)
        print('365-day signals complete',len(signals),flush=True)
    if args.stage != 'signals':
        from replay_v22_365d import run
        run(args.output)


if __name__=='__main__':
    main()
