"""One-click end-to-end demo runner for FD-CATE."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .artifacts import write_artifacts
from .benchmark import run_quick_benchmark, save_benchmark_report
from .estimator import FDCATE
from .io import from_dataframe


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_demo(
    *,
    outdir: str | Path,
    n: int = 120,
    d: int = 6,
    seed: int = 2026,
    method: str = "fd-dr",
    nuisance_learner: str = "xgb",
    run_benchmark: bool = True,
    fd_r_b_learner: str = "xgb",
    fd_r_g_solver: str = "direct",
    fd_r_swap_average: bool = True,
) -> Dict[str, Any]:
    """Run synthetic generation -> fit -> optional benchmark in one command."""
    from FDCATE import simulate_fd_data_md

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    synthetic_path = outdir_path / "synthetic.csv"
    fit_out = outdir_path / "fit_out"

    # 1) synthetic dataset
    data = simulate_fd_data_md(n=n, d=d, seed=seed)
    cols = {f"x{i}": data.C[:, i] for i in range(data.C.shape[1])}
    cols["y"] = data.Y
    cols["t"] = data.X
    cols["m"] = data.Z
    pd.DataFrame(cols).to_csv(synthetic_path, index=False)

    # 2) fit + artifacts
    df = pd.read_csv(synthetic_path)
    X, y, t, m, schema = from_dataframe(
        df,
        outcome="y",
        treatment="t",
        mediator="m",
        covariates=None,
    )
    est = FDCATE(
        method=method,
        nuisance_learner=nuisance_learner,
        fd_r_b_learner=fd_r_b_learner,
        fd_r_g_solver=fd_r_g_solver,
        fd_r_swap_average=fd_r_swap_average,
        random_state=seed,
        verbose=0,
    )
    est.fit(X, y, t=t, m=m)

    fit_out.mkdir(parents=True, exist_ok=True)
    model_path = fit_out / "model.pkl"
    est.save(model_path)
    effects = est.effect(X)
    write_artifacts(
        outdir=fit_out,
        estimator=est,
        effects=effects,
        input_spec={
            "n": int(X.shape[0]),
            "d": int(X.shape[1]),
            "treatment_type": "binary",
            "mediator_type": "binary",
            "outcome_type": "continuous_or_binary",
            "schema": schema,
        },
        command_line="fdcate demo",
        data_hash=_sha256_of_file(synthetic_path),
    )

    benchmark_path: Path | None = None
    if run_benchmark:
        benchmark_path = outdir_path / "benchmark_quick.json"
        report = run_quick_benchmark(
            n=max(60, min(120, n)),
            d=min(6, d),
            seed=seed,
            learner=nuisance_learner,
            fd_r_g_solver=fd_r_g_solver,
            fd_r_swap_average=fd_r_swap_average,
            fd_r_b_learner=fd_r_b_learner,
        )
        save_benchmark_report(report, benchmark_path)

    return {
        "outdir": outdir_path,
        "synthetic": synthetic_path,
        "fit_out": fit_out,
        "ate": float(est.ate_),
        "artifacts": [
            fit_out / "summary.txt",
            fit_out / "results.json",
            fit_out / "diagnostics.json",
            fit_out / "effects.csv",
            fit_out / "model.pkl",
        ],
        "benchmark": benchmark_path,
    }

