"""Independent causal daily input completeness for foundation model inference."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_crypto_rl_top10 import sha


def main():
    output = ROOT / "results/backtests/crypto_foundation_models_20261024"
    quality = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "context_days": 128, "feature_readiness_days": 31, "variants": {},
               "independent_from_model_adapter": True, "prices_interpolated": False}
    for variant, directory in (("original", "crypto_rl_top10"), ("ex_ada", "crypto_rl_top10_ex_ada")):
        path = ROOT / "data" / directory / "dataset.pkl.gz"
        data = pd.read_pickle(path)
        frames = {field: data["hourly_" + field] for field in ("open", "high", "low", "close", "volume")}
        index = frames["open"].index
        if not index.equals(pd.date_range(index[0], index[-1], freq="h")):
            raise AssertionError("The common UTC hourly grid is incomplete")
        complete = pd.DataFrame(True, index=index, columns=frames["open"].columns)
        for name, frame in frames.items():
            if not frame.index.equals(index) or not frame.columns.equals(complete.columns):
                raise AssertionError("Five hourly matrices are not exactly aligned")
            valid = np.isfinite(frame) & (frame.ge(0) if name == "volume" else frame.gt(0))
            complete &= valid
        daily_complete = complete.resample("D").sum().eq(24)
        daily_volume = frames["volume"].resample("D").sum(min_count=24).where(daily_complete)
        volume_ready = daily_volume.rolling(30, min_periods=30).mean().gt(0)
        ready31 = daily_complete.rolling(31, min_periods=31).sum().eq(31) & volume_ready
        ready128 = daily_complete.rolling(128, min_periods=128).sum().eq(128) & volume_ready
        ready31.index = ready31.index + pd.Timedelta(days=1)
        ready128.index = ready128.index + pd.Timedelta(days=1)
        universe = data["universe"].copy()
        universe["date"] = pd.to_datetime(universe.date, utc=True)
        universe["asset_key"] = universe.asset_key.astype(str)
        details = {}
        for split, start, end in (("validation", "2025-07-25", "2025-10-23"),
                                   ("test", "2025-10-23", "2026-10-23")):
            selected = universe.loc[universe.date.ge(pd.Timestamp(start, tz="UTC")) & universe.date.lt(pd.Timestamp(end, tz="UTC"))]
            not_ready31, extra = [], []
            eligible31 = eligible128 = 0
            for row in selected.itertuples(index=False):
                k, day = str(row.asset_key), row.date
                r31 = bool(ready31.loc[day, k])
                r128 = bool(ready128.loc[day, k])
                eligible31 += r31
                eligible128 += r128
                item = {"date": day.isoformat(), "asset_key": k, "symbol": row.symbol}
                if not r31:
                    not_ready31.append(item)
                elif not r128:
                    extra.append(item)
            details[split] = {"slot_days": len(selected), "ready31_slot_days": eligible31,
                              "ready128_slot_days": eligible128, "extra_context128_excluded_count": len(extra),
                              "extra_context128_excluded": extra, "existing_ready31_excluded": not_ready31}
        quality["variants"][variant] = {"dataset_sha256": sha(path), "split_counts": details}
    target = output / "sources/independent_input_preflight.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({variant: {split: {key: value for key, value in row.items()
                                      if key.endswith("count") or key.endswith("slot_days")}
                               for split, row in body["split_counts"].items()}
                      for variant, body in quality["variants"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
