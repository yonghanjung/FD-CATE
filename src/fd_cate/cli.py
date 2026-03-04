"""Command-line interface for fd-cate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .artifacts import write_artifacts
from .benchmark import run_quick_benchmark, save_benchmark_report
from .diagnostics import compute_diagnostics
from .estimator import FDCATE
from .io import from_dataframe


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_covariates(arg: str | None) -> list[str] | None:
    if arg is None or not arg.strip():
        return None
    return [x.strip() for x in arg.split(",") if x.strip()]


def cmd_fit(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    df = pd.read_csv(data_path)
    covariates = _parse_covariates(args.covariates)
    X, y, t, m, schema = from_dataframe(
        df,
        outcome=args.outcome,
        treatment=args.treat,
        mediator=args.med,
        covariates=covariates,
    )

    est = FDCATE(
        method=args.method,
        nuisance_learner=args.nuisance_learner,
        random_state=args.random_state,
        verbose=int(args.verbose),
    )
    est.fit(X, y, t=t, m=m)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "model.pkl"
    est.save(model_path)

    effects = est.effect(X)
    input_spec = {
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "treatment_type": "binary",
        "mediator_type": "binary",
        "outcome_type": "continuous_or_binary",
        "schema": schema,
    }
    write_artifacts(
        outdir=outdir,
        estimator=est,
        effects=effects,
        input_spec=input_spec,
        command_line="fdcate fit",
        data_hash=_sha256_of_file(data_path),
    )

    print(f"ATE={est.ate_:.6f}")
    print(f"Saved artifacts to: {outdir}")
    return 0


def cmd_effect(args: argparse.Namespace) -> int:
    est = FDCATE.load(args.model)
    df = pd.read_csv(args.data)

    if args.covariates:
        covariates = _parse_covariates(args.covariates)
        X_new = df.loc[:, covariates].apply(pd.to_numeric, errors="coerce").dropna(axis=0).to_numpy(dtype=float)
    else:
        X_new = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0).to_numpy(dtype=float)

    tau = est.effect(X_new, t0=args.t0, t1=args.t1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"tau": tau}).to_csv(out, index=False)
    print(f"Saved effects to: {out}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.data)
    covariates = _parse_covariates(args.covariates)
    X, y, t, m, schema = from_dataframe(
        df,
        outcome=args.outcome,
        treatment=args.treat,
        mediator=args.med,
        covariates=covariates,
    )
    diagnostics = compute_diagnostics(X, y, t, m, random_state=args.random_state)
    payload = {
        "schema_name": "fdcate.diagnostics",
        "schema_version": 0,
        "input_schema": schema,
        "diagnostics": diagnostics,
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_synthetic(args: argparse.Namespace) -> int:
    from FDCATE import simulate_fd_data_md

    data = simulate_fd_data_md(n=args.n, d=args.d, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = {f"x{i}": data.C[:, i] for i in range(data.C.shape[1])}
    cols["y"] = data.Y
    cols["t"] = data.X
    cols["m"] = data.Z
    pd.DataFrame(cols).to_csv(out, index=False)
    print(f"Saved synthetic dataset to: {out}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    report = run_quick_benchmark(
        n=args.n,
        d=args.d,
        seed=args.seed,
        learner=args.nuisance_learner,
    )
    print(json.dumps(report["results"], indent=2))

    if args.out:
        out_path = save_benchmark_report(report, args.out)
        print(f"Saved benchmark report to: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fdcate", description="FD-CATE command line interface")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="fit model and write standard artifacts")
    p_fit.add_argument("--data", required=True, help="Path to CSV data")
    p_fit.add_argument("--outcome", required=True, help="Outcome column name")
    p_fit.add_argument("--treat", required=True, help="Treatment column name")
    p_fit.add_argument("--med", required=True, help="Mediator column name")
    p_fit.add_argument(
        "--covariates",
        default=None,
        help="Comma-separated covariate columns. If omitted, all remaining columns are used.",
    )
    p_fit.add_argument("--method", default="fd-dr", choices=["fd-dr", "fd-r", "fd-pi"])
    p_fit.add_argument("--nuisance-learner", default="xgb", choices=["xgb", "nn"])
    p_fit.add_argument("--random-state", type=int, default=0)
    p_fit.add_argument("--verbose", type=int, default=0)
    p_fit.add_argument("--outdir", required=True, help="Output artifact directory")
    p_fit.set_defaults(func=cmd_fit)

    p_effect = sub.add_parser("effect", help="load model and compute effects")
    p_effect.add_argument("--model", required=True, help="Path to model.pkl")
    p_effect.add_argument("--data", required=True, help="Path to CSV with covariates")
    p_effect.add_argument(
        "--covariates",
        default=None,
        help="Comma-separated covariates. If omitted, all numeric columns are used.",
    )
    p_effect.add_argument("--t0", type=float, default=0.0)
    p_effect.add_argument("--t1", type=float, default=1.0)
    p_effect.add_argument("--out", default="effects.csv", help="Output CSV path")
    p_effect.set_defaults(func=cmd_effect)

    p_doctor = sub.add_parser("doctor", help="run diagnostics checks")
    p_doctor.add_argument("--data", required=True)
    p_doctor.add_argument("--outcome", required=True)
    p_doctor.add_argument("--treat", required=True)
    p_doctor.add_argument("--med", required=True)
    p_doctor.add_argument(
        "--covariates",
        default=None,
        help="Comma-separated covariate columns. If omitted, all remaining columns are used.",
    )
    p_doctor.add_argument("--random-state", type=int, default=0)
    p_doctor.set_defaults(func=cmd_doctor)

    p_syn = sub.add_parser("synthetic", help="create a toy synthetic dataset")
    p_syn.add_argument("--n", type=int, default=200)
    p_syn.add_argument("--d", type=int, default=10)
    p_syn.add_argument("--seed", type=int, default=42)
    p_syn.add_argument("--out", default="synthetic.csv")
    p_syn.set_defaults(func=cmd_synthetic)

    p_bench = sub.add_parser("benchmark", help="run quick deterministic benchmark")
    p_bench.add_argument("--n", type=int, default=120)
    p_bench.add_argument("--d", type=int, default=6)
    p_bench.add_argument("--seed", type=int, default=2026)
    p_bench.add_argument("--nuisance-learner", default="xgb", choices=["xgb", "nn"])
    p_bench.add_argument(
        "--out",
        default="results/benchmark_quick.json",
        help="Output path for benchmark JSON report. Set empty string to skip saving.",
    )
    p_bench.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
