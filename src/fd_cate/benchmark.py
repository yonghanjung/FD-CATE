"""Deterministic benchmark helpers for FD-CATE v0.x."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ._version import __version__


def _now_iso_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _as_float(value: Any) -> float:
    return float(np.asarray(value, dtype=float).item())


def _validate_choice(name: str, value: str, choices: set[str]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}")


def _compute_rmse_report(
    *,
    C: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    tau_true: np.ndarray,
    seed: int,
    learner: str,
    fd_r_g_solver: str,
    fd_r_swap_average: bool,
    fd_r_b_learner: str,
) -> Dict[str, float]:
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
        g_solver=fd_r_g_solver,
        swap_average=fd_r_swap_average,
        nuisance_learner=learner,
        b_learner=fd_r_b_learner,
    )

    return {
        "fd-pi_rmse": _as_float(rmse(np.asarray(tau_fd_pi), tau_true)),
        "fd-dr_rmse": _as_float(rmse(np.asarray(tau_fd_dr), tau_true)),
        "fd-r_rmse": _as_float(rmse(np.asarray(tau_fd_r), tau_true)),
    }


def run_quick_benchmark(
    *,
    n: int = 120,
    d: int = 6,
    seed: int = 2026,
    learner: str = "xgb",
    fd_r_g_solver: str = "direct",
    fd_r_swap_average: bool = True,
    fd_r_b_learner: str = "xgb",
) -> Dict[str, Any]:
    """Run a compact benchmark profile intended for CI/local regression checks."""
    _validate_choice("learner", learner, {"xgb", "nn"})
    _validate_choice("fd_r_g_solver", fd_r_g_solver, {"direct", "ratio"})
    _validate_choice("fd_r_b_learner", fd_r_b_learner, {"xgb", "nn"})

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
        fd_r_g_solver=fd_r_g_solver,
        fd_r_swap_average=fd_r_swap_average,
        fd_r_b_learner=fd_r_b_learner,
    )
    weak_report = _compute_rmse_report(
        C=weak.C,
        X=weak.X,
        Z=weak.Z,
        Y=weak.Y,
        tau_true=weak.tau_true,
        seed=seed + 200,
        learner=learner,
        fd_r_g_solver=fd_r_g_solver,
        fd_r_swap_average=fd_r_swap_average,
        fd_r_b_learner=fd_r_b_learner,
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
            "fd_r": {
                "g_solver": fd_r_g_solver,
                "swap_average": bool(fd_r_swap_average),
                "b_learner": fd_r_b_learner,
            },
        },
        "results": {
            "clean": clean_report,
            "weak-overlap": weak_report,
            "aggregate_mean_rmse": aggregate,
        },
    }


def _method_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run_multiseed_benchmark(
    *,
    n: int = 120,
    d: int = 6,
    seed: int = 2026,
    n_seeds: int = 10,
    learner: str = "xgb",
    fd_r_g_solver: str = "direct",
    fd_r_swap_average: bool = True,
    fd_r_b_learner: str = "xgb",
) -> Dict[str, Any]:
    """Run a multi-seed benchmark profile and return per-seed + summary stats."""
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")

    seeds = [int(seed + i) for i in range(int(n_seeds))]
    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        quick = run_quick_benchmark(
            n=n,
            d=d,
            seed=s,
            learner=learner,
            fd_r_g_solver=fd_r_g_solver,
            fd_r_swap_average=fd_r_swap_average,
            fd_r_b_learner=fd_r_b_learner,
        )
        per_seed.append(
            {
                "seed": s,
                "results": quick["results"],
            }
        )

    method_map = {
        "fd-pi": "fd-pi_rmse",
        "fd-dr": "fd-dr_rmse",
        "fd-r": "fd-r_rmse",
    }
    summary: Dict[str, Any] = {
        "clean": {},
        "weak-overlap": {},
        "aggregate_mean_rmse": {},
    }
    for method_short, method_long in method_map.items():
        clean_vals = [float(item["results"]["clean"][method_long]) for item in per_seed]
        weak_vals = [float(item["results"]["weak-overlap"][method_long]) for item in per_seed]
        agg_vals = [float(item["results"]["aggregate_mean_rmse"][method_short]) for item in per_seed]
        summary["clean"][method_short] = _method_stats(clean_vals)
        summary["weak-overlap"][method_short] = _method_stats(weak_vals)
        summary["aggregate_mean_rmse"][method_short] = _method_stats(agg_vals)

    return {
        "schema_name": "fdcate.benchmark",
        "schema_version": 0,
        "package_version": __version__,
        "created_at": _now_iso_local(),
        "config": {
            "profile": "multiseed",
            "n": int(n),
            "d": int(d),
            "seed_start": int(seed),
            "n_seeds": int(n_seeds),
            "seeds": seeds,
            "learner": learner,
            "scenarios": ["clean", "weak-overlap"],
            "fd_r": {
                "g_solver": fd_r_g_solver,
                "swap_average": bool(fd_r_swap_average),
                "b_learner": fd_r_b_learner,
            },
        },
        "results": {
            "per_seed": per_seed,
            "summary": summary,
        },
    }


def save_benchmark_report(report: Dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path
