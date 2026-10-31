"""Build a separate historical Top10 dataset excluding ADA (CMC UCID 2010).

The parent builder is loaded into a private module instance and its exclusion
hook is extended in memory only. Parent sources, caches and results are read
only. Evidence reuse is by independent physical copies, never hard links.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARENT_CACHE = ROOT / "data/crypto_rl_top10"
PARENT_OUTPUT = ROOT / "results/backtests/crypto_rl_top10_20261024"
DEFAULT_CACHE = ROOT / "data/crypto_rl_top10_ex_ada"
DEFAULT_OUTPUT = ROOT / "results/backtests/crypto_rl_ex_ada_20261024"
EXCLUDED_ASSET_KEY = "2010"
REUSE_DIRECTORIES = ("snapshots", "probes", "first_hour", "raw", "patches")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _guard_paths(cache: Path, output: Path) -> None:
    protected = (PARENT_CACHE.resolve(), PARENT_OUTPUT.resolve())
    for destination in (cache, output):
        if any(destination == p or destination.is_relative_to(p) or p.is_relative_to(destination) for p in protected):
            raise ValueError(f"New study must not overlap the parent study: {destination}")
    if cache == output or cache.is_relative_to(output) or output.is_relative_to(cache):
        raise ValueError("Cache and result directories must be separate")


def _integrity_snapshot() -> dict:
    files = sorted(p for p in PARENT_CACHE.rglob("*") if p.is_file())
    extras = [ROOT / "scripts/crypto_rl_data.py"] + [PARENT_OUTPUT / "sources" / name for name in
              ("data_manifest.json", "universe_daily.csv", "asset_registry.json")]
    files += extras
    def record(path: Path) -> dict:
        if path.is_symlink():
            raise ValueError(f"Do not follow parent evidence symlinks: {path}")
        stat = path.stat()
        return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": _sha(path)}
    with ThreadPoolExecutor(max_workers=6) as pool:
        entries = list(pool.map(record, files))
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {"parent_cache": str(PARENT_CACHE.resolve()), "protected_file_count": len(entries),
            "total_bytes": sum(x["bytes"] for x in entries), "entries_sha256": hashlib.sha256(canonical).hexdigest(),
            "entries": entries}


def _assert_parent_unchanged(before: dict, output: Path) -> dict:
    after = _integrity_snapshot()
    old = {x["path"]: x for x in before["entries"]}
    new = {x["path"]: x for x in after["entries"]}
    changed = [p for p in sorted(set(old) | set(new)) if old.get(p) != new.get(p)]
    evidence = {"checked_at": datetime.now(timezone.utc).isoformat(), "passed": not changed,
                "protected_file_count": len(old), "before_entries_sha256": before["entries_sha256"],
                "after_entries_sha256": after["entries_sha256"], "changed_paths": changed,
                "comparison": "full file inventory, byte size, SHA256 and mtime_ns; access times ignored"}
    _write(output / "sources/parent_integrity_check.json", evidence)
    if changed:
        raise RuntimeError(f"Parent study changed during independent build: {changed[:5]}")
    return evidence


def _copy_evidence(cache: Path, output: Path) -> dict:
    jobs = []
    for directory in REUSE_DIRECTORIES:
        source = PARENT_CACHE / directory
        jobs.extend((p, cache / p.relative_to(PARENT_CACHE)) for p in source.rglob("*") if p.is_file())
    robots = PARENT_CACHE / "robots.txt"
    if robots.exists():
        jobs.append((robots, cache / "robots.txt"))
    def copy(job: tuple[Path, Path]) -> dict:
        source, destination = job
        if source.is_symlink():
            raise ValueError(f"Do not reuse symlink: {source}")
        expected = _sha(source)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if os.path.samefile(source, destination) or _sha(destination) != expected:
            raise RuntimeError(f"Evidence copy must be independent and byte-identical: {destination}")
        return {"source": str(source), "destination": str(destination), "sha256": expected,
                "bytes": source.stat().st_size, "samefile": False}
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(copy, jobs))
    manifest = {"method": "independent physical copy; no hardlink, symlink or shared writable fallback",
                "files": records, "file_count": len(records), "bytes": sum(x["bytes"] for x in records)}
    _write(output / "sources/reused_evidence.json", manifest)
    return manifest


def _load_private_builder():
    path = ROOT / "scripts/crypto_rl_data.py"
    name = "_crypto_rl_ex_ada_private_builder"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the frozen parent builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    parent_exclude = module._exclude
    def exclude(asset: dict) -> str | None:
        if str(asset["id"]) == EXCLUDED_ASSET_KEY:
            return "user-directed exclusion: ADA CMC2010; refill next historically ranked eligible asset"
        return parent_exclude(asset)
    module._exclude = exclude
    return module


def _selection_plan(base, cache: Path, output: Path, offline: bool) -> dict:
    anchors = sorted(set(pd.date_range(base.TRAIN_START, base.END, freq="MS", inclusive="left").tolist()
                         + [base.VAL_START, base.TEST_START]))
    dates = sorted({base._snapshot_date(t) for t in anchors})
    snapshots = {date: base._load_snapshot(date, cache, True) for date in dates}
    with ThreadPoolExecutor(max_workers=6) as pool:
        selected = list(pool.map(lambda t: base._select_anchor(t, snapshots, cache, offline), anchors))
    parent_selections = {pd.Timestamp(x["anchor"]): x for x in _read(PARENT_CACHE / "universe_selection.json")}
    parent_manifest = _read(PARENT_CACHE / "manifest.json")
    parent_registry = parent_manifest["registry"]
    registry, evidence, differences = {}, [], []
    for anchor, (chosen, proof) in zip(anchors, selected):
        keys = [r["asset_key"] for r in chosen]
        old_keys = parent_selections[anchor]["selected_asset_keys"]
        if EXCLUDED_ASSET_KEY in keys or len(keys) != 10 or len(set(keys)) != 10:
            raise AssertionError("ADA excluded and exactly ten distinct ranked assets required")
        for item in chosen:
            identity = item["_identity"]
            key = item["asset_key"]
            if key in registry and registry[key]["binance_symbol"] != identity["binance_symbol"]:
                raise ValueError(f"Unreviewed identity change for {key}")
            registry.setdefault(key, identity)
        added = [k for k in keys if k not in old_keys]
        removed = [k for k in old_keys if k not in keys]
        if (EXCLUDED_ASSET_KEY in old_keys and removed != [EXCLUDED_ASSET_KEY]) or (EXCLUDED_ASSET_KEY not in old_keys and (added or removed)):
            raise AssertionError(f"Exclusion should change membership only by replacing ADA: {anchor}")
        differences.append({"anchor": anchor.isoformat(), "snapshot_date": proof["snapshot"]["snapshot_date"],
                            "old_asset_keys": "|".join(old_keys), "new_asset_keys": "|".join(keys),
                            "old_symbols": "|".join(parent_registry[k]["symbol"] for k in old_keys),
                            "new_symbols": "|".join(r["symbol"] for r in chosen),
                            "removed_asset_keys": "|".join(removed), "removed_symbols": "|".join(parent_registry[k]["symbol"] for k in removed),
                            "added_asset_keys": "|".join(added), "added_symbols": "|".join(registry[k]["symbol"] for k in added),
                            "ada_previously_selected": EXCLUDED_ASSET_KEY in old_keys,
                            "members_changed": bool(added or removed)})
        evidence.append(proof)
    new_keys = sorted(set(registry) - set(parent_registry), key=int)
    added_symbols = sorted({r["added_symbols"] for r in differences if r["added_symbols"]})
    plan = {"exclusion": {"asset_key": EXCLUDED_ASSET_KEY, "symbol": "ADA", "rule": "exclude before ranked eligibility scan; refill to ten"},
            "anchor_count": len(anchors), "changed_anchors": sum(x["members_changed"] for x in differences),
            "parent_asset_count": len(parent_registry), "new_asset_count": len(registry),
            "new_union_assets": [registry[k] for k in new_keys], "replacement_symbols": added_symbols,
            "estimated_max_extra_hour_rows": len(new_keys) * len(pd.date_range(base.DATA_START, base.END, freq="h", inclusive="left")),
            "estimated_max_extra_archive_pairs": len(new_keys) * 69,
            "estimate_note": "Upper bound using all 47 completed months plus 22 recent daily archives; actual listing may reduce it.",
            "selection_evidence": evidence, "registry": registry}
    _write(cache / "universe_selection.json", evidence)
    _write(output / "sources/selection_plan.json", plan)
    pd.DataFrame(differences).to_csv(output / "sources/pool_anchor_changes.csv", index=False)
    print(json.dumps({k: v for k, v in plan.items() if k not in ("selection_evidence", "registry")}, ensure_ascii=False), flush=True)
    return plan


def _compare_shared_prices(payload: dict) -> dict:
    parent_path = PARENT_CACHE / "dataset.pkl.gz"
    parent_manifest = _read(PARENT_CACHE / "manifest.json")
    if _sha(parent_path) != parent_manifest["dataset_sha256"]:
        raise RuntimeError("Parent dataset does not match its frozen external manifest")
    parent = pickle.loads(gzip.decompress(parent_path.read_bytes()))
    common = sorted(set(parent["assets"]) & set(payload["assets"]), key=int)
    checks = {}
    for name in ("hourly_open", "hourly_high", "hourly_low", "hourly_close", "hourly_volume"):
        old, new = parent[name].loc[:, common], payload[name].loc[:, common]
        if not old.index.equals(new.index) or not np.array_equal(old.to_numpy(), new.to_numpy(), equal_nan=True):
            raise RuntimeError(f"Common asset values or NaN positions changed: {name}")
        checks[name] = {"passed": True, "hours": len(old), "assets": len(common),
                        "compared_cells": old.size, "nan_cells": int(old.isna().sum().sum())}
    return {"parent_dataset_sha256": parent_manifest["dataset_sha256"], "common_asset_keys": common,
            "comparison": "exact float values and NaN positions; no tolerance or fill", "matrices": checks}


def build_dataset(cache_dir: str | Path = DEFAULT_CACHE, output_dir: str | Path = DEFAULT_OUTPUT,
                  offline: bool = False, selection_only: bool = False) -> dict:
    cache, output = Path(cache_dir).resolve(), Path(output_dir).resolve()
    _guard_paths(cache, output)
    cache.mkdir(parents=True, exist_ok=True)
    (output / "sources").mkdir(parents=True, exist_ok=True)
    before_path = output / "sources/parent_integrity_before.json"
    current = _integrity_snapshot()
    if before_path.exists():
        before = _read(before_path)
        if current["entries"] != before["entries"]:
            raise RuntimeError("Parent inputs changed since the independent study began")
    else:
        before = current
        _write(before_path, before)
    _copy_evidence(cache, output)
    base = _load_private_builder()
    plan = _selection_plan(base, cache, output, offline)
    if selection_only:
        _assert_parent_unchanged(before, output)
        return plan
    payload = base.build_dataset(cache, output, offline)
    if [x["selected_asset_keys"] for x in plan["selection_evidence"]] != [x["selected_asset_keys"] for x in payload["manifest"]["selection_evidence"]]:
        raise RuntimeError("Full build universe differs from its preregistered exclusion plan")
    shared = _compare_shared_prices(payload)
    integrity = _assert_parent_unchanged(before, output)
    manifest = payload["manifest"]
    manifest.pop("dataset_sha256", None)
    manifest.update({"study": "historical dynamic Top10 excluding ADA UCID2010; next eligible ranked member refills each anchor",
                     "exclusion": plan["exclusion"], "parent_dataset": {"path": str(PARENT_CACHE / "dataset.pkl.gz"),
                       "sha256": shared["parent_dataset_sha256"]}, "parent_integrity": integrity,
                     "adapter_source_sha256": _sha(Path(__file__)), "parent_builder_source_sha256": _sha(ROOT / "scripts/crypto_rl_data.py"),
                     "common_asset_price_comparison": shared, "new_union_assets": plan["new_union_assets"],
                     "dataset_path": str(cache / "dataset.pkl.gz"),
                     "embedded_manifest_hash_note": "Dataset SHA256 is authoritative in external manifest.json; not recursively embedded."})
    encoded = gzip.compress(pickle.dumps(payload, protocol=5), mtime=0)
    (cache / "dataset.pkl.gz").write_bytes(encoded)
    manifest["dataset_sha256"] = hashlib.sha256(encoded).hexdigest()
    _write(cache / "manifest.json", manifest)
    _write(output / "sources/data_manifest.json", manifest)
    summary = {"exclusion": plan["exclusion"], "anchors": plan["anchor_count"], "changed_anchors": plan["changed_anchors"],
               "replacement_symbols": plan["replacement_symbols"], "new_union_assets": plan["new_union_assets"],
               "parent_asset_count": plan["parent_asset_count"], "asset_count": len(payload["assets"]),
               "dataset_sha256": manifest["dataset_sha256"], "shared_prices": shared, "parent_integrity": integrity,
               "coverage": manifest["coverage"], "archive_pairs": len(manifest["archive_evidence"])}
    _write(output / "sources/data_change_summary.json", summary)
    print(f"EX_ADA_DATA_READY {cache / 'dataset.pkl.gz'} SHA256={manifest['dataset_sha256']}", flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    build_dataset(args.cache_dir, args.output_dir, args.offline, args.selection_only)
