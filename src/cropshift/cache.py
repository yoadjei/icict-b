"""Resumable pipeline steps: a result is saved atomically when it is computed and loaded on rerun.

An interrupted script therefore resumes from its last finished step. Every step seeds its own
random numbers, so resumed and uninterrupted runs give the same results.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _write(frame: pd.DataFrame, path: Path, suffix: str) -> None:
    """Write in the format of `suffix` (the final file's suffix, not the temporary file's)."""
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def cached_frame(path: Path, compute: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Load `path` if it exists; otherwise compute the frame, save it atomically and return it."""
    if path.exists():
        return _read(path)
    frame = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    _write(frame, tmp, path.suffix)
    os.replace(tmp, path)
    return frame
