"""Data loading helpers used by API and CLI."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ArrayTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]


def _as_1d_float(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}.")
    return arr.astype(float)


def _as_2d_float(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
    return arr.astype(float)


def _ensure_binary(name: str, values: np.ndarray) -> np.ndarray:
    unique = np.unique(values[~np.isnan(values)])
    if not set(unique.tolist()).issubset({0.0, 1.0}):
        raise ValueError(
            f"{name} must be binary {{0,1}} for v0.1. Found values: {unique.tolist()}"
        )
    return values.astype(int)


def from_dataframe(
    df: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    mediator: str,
    covariates: Optional[Sequence[str]] = None,
) -> ArrayTuple:
    """Extract (X, y, t, m) arrays from a DataFrame with a deterministic schema."""
    required = {outcome, treatment, mediator}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if covariates is None:
        covariates = [
            c for c in df.columns if c not in {outcome, treatment, mediator}
        ]
    else:
        covariates = list(covariates)

    if not covariates:
        raise ValueError("At least one covariate is required.")

    for col in covariates:
        if col not in df.columns:
            raise ValueError(f"Covariate column '{col}' not found in dataframe.")

    cols = [*covariates, outcome, treatment, mediator]
    numeric = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    n_before = len(numeric)
    clean = numeric.dropna(axis=0)
    n_after = len(clean)

    X = _as_2d_float("X", clean.loc[:, covariates].to_numpy())
    y = _as_1d_float("y", clean.loc[:, outcome].to_numpy())
    t = _ensure_binary("treatment", _as_1d_float("t", clean.loc[:, treatment].to_numpy()))
    m = _ensure_binary("mediator", _as_1d_float("m", clean.loc[:, mediator].to_numpy()))

    schema = {
        "outcome": outcome,
        "treatment": treatment,
        "mediator": mediator,
        "covariates": list(covariates),
        "n_before_dropna": int(n_before),
        "n_after_dropna": int(n_after),
    }
    return X, y, t, m, schema
