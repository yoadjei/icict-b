"""Extract frozen Prithvi-EO and ViT-Base CLS embeddings for every field in the feature table.

Usage:
    python scripts/04_extract_embeddings.py --model prithvi --threads 3
    python scripts/04_extract_embeddings.py --model vit --threads 3

Resumable: rows already in the output parquet are skipped.
Outputs (gitignored): data/processed/embeddings_<model>.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cropshift import embeddings as emb  # noqa: E402

FEATURES = ROOT / "data" / "processed" / "master_features.parquet"
FIELDS = ROOT / "data" / "interim" / "fields"
WEIGHTS = ROOT / "models" / "Prithvi_EO_V1_100M.pt"
BATCH = 16
SAVE_EVERY = 20  # batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["prithvi", "vit"], required=True)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    out = ROOT / "data" / "processed" / f"embeddings_{args.model}.parquet"
    ids = pd.read_parquet(FEATURES, columns=["field_id"])["field_id"].tolist()
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    finished = set(done["field_id"]) if len(done) else set()
    todo = [f for f in ids if f not in finished]
    print(f"{args.model}: {len(ids)} fields, {len(finished)} done, {len(todo)} to do", flush=True)

    if args.model == "prithvi":
        model, to_input, encode = emb.load_prithvi(WEIGHTS), emb.prithvi_input, emb.prithvi_cls
    else:
        model, to_input, encode = emb.load_vit(), emb.vit_input, emb.vit_cls

    records, start = [], time.time()
    for b, i in enumerate(range(0, len(todo), BATCH)):
        chunk = todo[i:i + BATCH]
        tensors = []
        for fid in chunk:
            with np.load(FIELDS / f"{fid}.npz") as z:
                tensors.append(to_input(emb.fill_masked(z["bands"], z["valid"])))
        vectors = encode(model, torch.stack(tensors))
        for fid, vec in zip(chunk, vectors):
            records.append({"field_id": fid, **{f"{args.model}_{j}": float(v) for j, v in enumerate(vec)}})
        if (b + 1) % SAVE_EVERY == 0 or i + BATCH >= len(todo):
            done = pd.concat([done, pd.DataFrame(records)], ignore_index=True)
            done.to_parquet(out, index=False)
            records = []
            rate = (time.time() - start) / (i + len(chunk))
            print(f"{len(done)}/{len(ids)} saved; {rate:.2f} s/field; eta {rate * (len(todo) - i - len(chunk)) / 60:.1f} min",
                  flush=True)
    print(f"finished {args.model}: {len(done)} rows in {out}", flush=True)


if __name__ == "__main__":
    main()
