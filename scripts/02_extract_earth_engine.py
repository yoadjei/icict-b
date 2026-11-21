"""Extract Sentinel-2 patches and CHIRPS series for every sampled field.

Usage:
    python scripts/02_extract_earth_engine.py --limit 40          # throughput test
    python scripts/02_extract_earth_engine.py --workers 16        # full run, resumable

Outputs (gitignored):
    data/interim/fields/<field_id>.npz   bands, valid, n_scenes, chirps_ms, chirps_mm, window
    logs/earth_engine_failures.csv        fields that failed after all retries
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cropshift import earth_engine as gee  # noqa: E402

LABELS = ROOT / "data" / "processed" / "labels_sampled.parquet"
OUT_DIR = ROOT / "data" / "interim" / "fields"
FAIL_LOG = ROOT / "logs" / "earth_engine_failures.csv"
GEE_PROJECT = "gee-01-492802"
MAX_TRIES = 5


def extract_one(ee, row: pd.Series) -> str:
    out = OUT_DIR / f"{row['field_id']}.npz"
    if out.exists():
        return "cached"
    window = gee.season_window(row)
    for attempt in range(1, MAX_TRIES + 1):
        try:
            bands, valid, n_scenes = gee.fetch_s2_patch(ee, row["lon"], row["lat"], window)
            chirps_ms, chirps_mm = gee.fetch_chirps(ee, row["lon"], row["lat"], window)
            tmp = out.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp, bands=bands, valid=valid, n_scenes=n_scenes, chirps_ms=chirps_ms, chirps_mm=chirps_mm,
                window_start=window.start.isoformat(), window_end=window.end.isoformat(),
                window_source=window.source)
            tmp.replace(out)
            return "ok"
        except Exception as exc:  # network, quota or server errors: back off and retry
            if attempt == MAX_TRIES:
                raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
            time.sleep(min(60, 2 ** attempt + random.random()))
    return "unreachable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    import ee
    ee.Initialize(project=GEE_PROJECT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(LABELS)
    if args.limit:
        labels = labels.groupby("country").sample(n=max(1, args.limit // 5), random_state=0)
    rows = [r for _, r in labels.iterrows()]
    print(f"fields to process: {len(rows)}  workers: {args.workers}", flush=True)

    start, done, failures, status_counts = time.time(), 0, [], {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_one, ee, r): r["field_id"] for r in rows}
        for fut in as_completed(futures):
            done += 1
            try:
                status = fut.result()
            except Exception as exc:
                status = "failed"
                failures.append({"field_id": futures[fut], "error": str(exc)[:300]})
            status_counts[status] = status_counts.get(status, 0) + 1
            if done % 100 == 0 or done == len(rows):
                rate = (time.time() - start) / done
                eta_min = rate * (len(rows) - done) / 60
                print(f"{done}/{len(rows)} {status_counts} {rate:.2f} s/field  eta {eta_min:.1f} min", flush=True)
    pd.DataFrame(failures, columns=["field_id", "error"]).to_csv(FAIL_LOG, index=False)
    print(f"finished in {(time.time() - start) / 60:.1f} min; {status_counts}; failures logged to {FAIL_LOG}", flush=True)


if __name__ == "__main__":
    main()
