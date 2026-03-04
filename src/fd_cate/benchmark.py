"""Deterministic benchmark helpers for FD-CATE v0.x."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ._version import __version__


def _now_iso_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _as_float(value: Any) -> float:
    return float(np.asarray(value, dtype=float).item())


def _compute_rmse_report(*, C: np.ndarray, X: np.ndarray, Z: np.ndarray, Y: np.ndarray, tau_true: np.ndarray, seed: int, learner: str) -> Dict[str, float]:
    from FDCATE import (
        fit_folds,
        rmse,
        tau_fd_dr_oof,
        tau_fd_r_3way_oof_smoothed,
        tau_naive_oof,
    )

    folds = fit_folds(C, X, Z, Y, seed, learner=learner)
    tau_fd_pi = tau_naive_oof(C, X, Z, Y, folds, 0.0, None, None, seed + 11)
    tau_fd_dr, _ = tau_fd_dr_oof(C, X, Z, Y, folds, 0.0, None, None, seed + 22)
    tau_fd_r, _ = tau_fd_r_3way_oof_smoothed(
        C,
        X,
        Z,
        Y,
        0.0,
        None,
        None,
        seed + 33,
        nuisance_learner=learner,
    )

    return {
        "fd-pi_rmse": _as_float(rmse(np.asarray(tau_fd_pi), tau_true)),
        "fd-dr_rmse": _as_float(rmse(np.asarray(tau_fd_dr), tau_true)),
        "fd-r_rmse": _as_float(rmse(np.asarray(tau_fd_r), tau_true)),
    }


def run_quick_benchmark(*, n: int = 120, d: int = 6, seed: int = 2026, learner: str = "xgb") -> Dict[str, Any]:
    """Run a compact benchmark profile intended for CI/local regression checks."""
    if learner not in {"xgb", "nn"}:
        raise ValueError("learner must be one of {'xgb', 'nn'}")

    from FDCATE import simulate_fd_data_md, simulate_fd_data_weak_overlap

    clean = simulate_fd_data_md(n=n, d=d, seed=seed, mediator_confound=0.0)
    weak = simulate_fd_data_weak_overlap(n=n, d=d, seed=seed + 1, kappa_e=6.0, kappa_q=1.0)

    clean_report = _compute_rmse_report(
        C=clean.C,
        X=clean.X,
        Z=clean.Z,
        Y=clean.Y,
        tau_true=clean.tau_true,
        seed=seed + 100,
        learner=learner,
    )
    weak_report = _compute_rmse_report(
        C=weak.C,
        X=weak.X,
        Z=weak.Z,
        Y=weak.Y,
        tau_true=weak.tau_true,
        seed=seed + 200,
        learner=learner,
    )

    methods = ("fd-pi_rmse", "fd-dr_rmse", "fd-r_rmse")
    aggregate = {
        method.replace("_rmse", ""): float(np.mean([clean_report[method], weak_report[method]]))
        for method in methods
    }

    return {
        "schema_name": "fdcate.benchmark",
        "schema_version": 0,
        "package_version": __version__,
        "created_at": _now_iso_local(),
        "config": {
            "profile": "quick",
            "n": int(n),
            "d": int(d),
            "seed": int(seed),
            "learner": learner,
            "scenarios": ["clean", "weak-overlap"],
        },
        "results": {
            "clean": clean_report,
            "weak-overlap": weak_report,
            "aggregate_mean_rmse": aggregate,
        },
    }


def save_benchmark_report(report: Dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path
