"""Dataset selection, validation, and safe user input parsing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .constants import (
    FEATURES, NUMERIC_FEATURES, LIMITS, TARGET, TYPE_OPTIONS,
    OFFICIAL_DATA, SIMULATED_DATA, ROOT,
)


class InputValidationError(ValueError):
    """Presented to the operator with an actionable message."""


def dataset_path() -> Path:
    if OFFICIAL_DATA.is_file():
        return OFFICIAL_DATA
    if SIMULATED_DATA.is_file():
        return SIMULATED_DATA
    raise FileNotFoundError("No training dataset found. Run bootstrap.py first.")


def dataset_kind(path: Path | None = None) -> str:
    return "official_uci_synthetic" if (path or dataset_path()).resolve() == OFFICIAL_DATA.resolve() else "locally_simulated"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_measurements(frame: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise InputValidationError("Provide at least one complete equipment record.")
    if len(frame) > max_rows:
        raise InputValidationError(f"Upload at most {max_rows:,} rows at a time.")
    missing = [col for col in FEATURES if col not in frame.columns]
    if missing:
        raise InputValidationError("Missing required column(s): " + ", ".join(missing))
    out = frame.loc[:, FEATURES].copy()
    out["Type"] = out["Type"].astype("string").str.strip().str.upper()
    bad_type = ~out["Type"].isin(TYPE_OPTIONS)
    if bad_type.any():
        raise InputValidationError(f"Invalid product type at CSV row {bad_type.idxmax() + 2}: use L, M, or H.")
    for col in NUMERIC_FEATURES:
        try:
            out[col] = pd.to_numeric(out[col], errors="raise")
        except (TypeError, ValueError) as exc:
            raise InputValidationError(f"Column '{col}' must contain numbers: {exc}") from exc
        values = out[col].to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        lo, hi = LIMITS[col]
        bad = bad | (values < lo) | (values > hi)
        if bad.any():
            idx = int(np.where(bad)[0][0]) + 2
            raise InputValidationError(f"Invalid '{col}' at data row {idx}; allowed range {lo:g}–{hi:g}.")
    return out


def load_training_data(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    path = path or dataset_path()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if TARGET not in df:
        raise ValueError(f"Dataset missing target column {TARGET!r}")
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("Training target must contain only 0 and 1 and no missing values.")
    if df[TARGET].nunique() != 2:
        raise ValueError("Training data needs both failure and nonfailure examples.")
    clean = validate_measurements(df, max_rows=1000000)
    summary = {
        "dataset_kind": dataset_kind(path), "rows": len(df),
        "missing_cells": int(df[FEATURES + [TARGET]].isna().sum().sum()),
        "duplicate_feature_target_rows": int(df[FEATURES + [TARGET]].duplicated().sum()),
        "failure_rows": int(df[TARGET].sum()),
        "failure_rate": float(df[TARGET].mean()), "path": str(path.relative_to(ROOT)),
    }
    clean[TARGET] = df[TARGET].astype(int).to_numpy()
    return clean, summary


def source_label(kind: str) -> str:
    if kind == "official_uci_synthetic":
        return "UCI AI4I 2020 • published synthetic benchmark"
    return "Locally generated simulator • NOT UCI benchmark data"
