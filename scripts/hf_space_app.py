"""Draft Hugging Face Space app for FD-CATE.

Copy this file to the root of a Gradio Space as `app.py`.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from fd_cate.artifacts import write_artifacts
from fd_cate.benchmark import run_quick_benchmark, save_benchmark_report
from fd_cate.estimator import FDCATE
from fd_cate.io import from_dataframe


METHOD_MAP = {
    "FD-PI": "fd-pi",
    "FD-DR": "fd-dr",
    "FD-R": "fd-r",
}


def _make_canonical_example_dataframe(*, n: int = 120, d: int = 6, seed: int = 2026) -> pd.DataFrame:
    from FDCATE import simulate_fd_data_md

    data = simulate_fd_data_md(n=n, d=d, seed=seed)
    cols = {f"x{i}": data.C[:, i] for i in range(data.C.shape[1])}
    cols["y"] = data.Y
    cols["t"] = data.X
    cols["m"] = data.Z
    return pd.DataFrame(cols)


def _effects_histogram(effects_path: Path, method_label: str):
    df = pd.read_csv(effects_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["tau"], bins=30, color="#d9480f", edgecolor="white")
    ax.set_title(f"{method_label} estimated effect distribution")
    ax.set_xlabel("Estimated treatment effect")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def _benchmark_table(benchmark_path: Path | None) -> pd.DataFrame:
    if benchmark_path is None or not benchmark_path.exists():
        return pd.DataFrame(
            [{"scenario": "benchmark skipped", "fd-pi": None, "fd-dr": None, "fd-r": None}]
        )

    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    results = payload["results"]
    rows: list[dict[str, object]] = []
    for scenario in ("clean", "weak-overlap"):
        block = results[scenario]
        rows.append(
            {
                "scenario": scenario,
                "fd-pi": block["fd-pi_rmse"],
                "fd-dr": block["fd-dr_rmse"],
                "fd-r": block["fd-r_rmse"],
            }
        )
    agg = results["aggregate_mean_rmse"]
    rows.append(
        {
            "scenario": "aggregate_mean_rmse",
            "fd-pi": agg["fd-pi"],
            "fd-dr": agg["fd-dr"],
            "fd-r": agg["fd-r"],
        }
    )
    return pd.DataFrame(rows)


def _summary_markdown(
    result: dict,
    benchmark_table: pd.DataFrame,
    *,
    method_label: str,
    source_label: str,
) -> str:
    artifacts = "\n".join(f"- `{Path(path).name}`" for path in result["artifacts"])
    schema = result["schema"]
    lines = [
        "## Run summary",
        f"- Source: {source_label}",
        f"- Method: `{method_label}`",
        f"- Estimated ATE: `{result['ate']:.6f}`",
        f"- Outcome / treatment / mediator: `{schema['outcome']}` / `{schema['treatment']}` / `{schema['mediator']}`",
        f"- Covariates: `{', '.join(schema['covariates'])}`",
        f"- Rows used after numeric cleaning: `{schema['n_after_dropna']}` / `{schema['n_before_dropna']}`",
        f"- Output directory: `{result['outdir']}`",
        "",
        "Artifacts:",
        artifacts,
    ]
    if result["benchmark"] is not None and not benchmark_table.empty:
        best = benchmark_table[benchmark_table["scenario"] == "aggregate_mean_rmse"]
        if not best.empty:
            row = best.iloc[0]
            lines.extend(
                [
                    "",
                    "Quick benchmark aggregate RMSE:",
                    f"- FD-PI: `{row['fd-pi']:.4f}`",
                    f"- FD-DR: `{row['fd-dr']:.4f}`",
                    f"- FD-R: `{row['fd-r']:.4f}`",
                ]
            )
    return "\n".join(lines)


def _zip_artifacts(outdir: Path) -> str:
    archive_base = outdir.parent / f"{outdir.name}_artifacts"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=outdir)
    return archive_path


def _run_from_dataframe(
    *,
    outdir: Path,
    dataframe: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    covariates: list[str] | None,
    input_filename: str,
    command_line: str,
    random_state: int,
    method: str,
    nuisance_learner: str,
    run_benchmark: bool,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    input_path = outdir / input_filename
    fit_out = outdir / "fit_out"
    dataframe.to_csv(input_path, index=False)

    frame = pd.read_csv(input_path)
    X, y, t, m, schema = from_dataframe(
        frame,
        outcome=outcome,
        treatment=treatment,
        mediator=mediator,
        covariates=covariates,
    )
    est = FDCATE(
        method=method,
        nuisance_learner=nuisance_learner,
        fd_r_b_learner=nuisance_learner,
        random_state=int(random_state),
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
        command_line=command_line,
        data_hash=None,
    )

    benchmark_path: Path | None = None
    if run_benchmark:
        benchmark_path = outdir / "benchmark_quick.json"
        report = run_quick_benchmark(
            n=max(60, min(120, int(X.shape[0]))),
            d=min(6, int(X.shape[1])),
            seed=int(random_state),
            learner=nuisance_learner,
            fd_r_b_learner=nuisance_learner,
        )
        save_benchmark_report(report, benchmark_path)

    return {
        "outdir": outdir,
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
        "schema": schema,
    }


def _parse_covariates(covariates_text: str) -> list[str]:
    return [item.strip() for item in covariates_text.split(",") if item.strip()]


def _load_data_source(
    *,
    data_source: str,
    csv_path: str | None,
    outcome_col: str,
    treatment_col: str,
    mediator_col: str,
    covariates_text: str,
    n: int,
    d: int,
    seed: int,
) -> tuple[pd.DataFrame, list[str] | None, str, str]:
    covariates = _parse_covariates(covariates_text)
    if data_source == "Canonical example":
        frame = _make_canonical_example_dataframe(n=int(n), d=int(d), seed=int(seed))
        if not covariates:
            covariates = [f"x{i}" for i in range(int(d))]
        source_label = f"canonical synthetic front-door example (n={int(n)}, d={int(d)}, seed={int(seed)})"
        input_filename = "synthetic.csv"
        return frame, covariates, source_label, input_filename

    if not csv_path:
        raise gr.Error("Upload a CSV file or switch the data source to Canonical example.")

    frame = pd.read_csv(csv_path)
    if not covariates:
        excluded = {outcome_col, treatment_col, mediator_col}
        covariates = [col for col in frame.columns if col not in excluded]
    source_label = f"uploaded CSV `{Path(csv_path).name}`"
    return frame, covariates, source_label, Path(csv_path).name


def run_space_demo(
    data_source: str,
    csv_path: str | None,
    outcome_col: str,
    treatment_col: str,
    mediator_col: str,
    covariates_text: str,
    n: int,
    d: int,
    method_label: str,
    nuisance_learner: str,
    seed: int,
    include_benchmark: bool,
):
    frame, covariates, source_label, input_filename = _load_data_source(
        data_source=data_source,
        csv_path=csv_path,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        mediator_col=mediator_col,
        covariates_text=covariates_text,
        n=int(n),
        d=int(d),
        seed=int(seed),
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="fdcate-space-"))
    result = _run_from_dataframe(
        outdir=tmpdir / "fdcate-demo",
        dataframe=frame,
        outcome=outcome_col,
        treatment=treatment_col,
        mediator=mediator_col,
        covariates=covariates,
        input_filename=input_filename,
        command_line="fdcate demo (HF Space)",
        random_state=int(seed),
        method=METHOD_MAP[method_label],
        nuisance_learner=nuisance_learner,
        run_benchmark=bool(include_benchmark),
    )

    effects_path = result["fit_out"] / "effects.csv"
    hist = _effects_histogram(effects_path, method_label)
    benchmark_df = _benchmark_table(result["benchmark"])
    summary = _summary_markdown(
        result,
        benchmark_df,
        method_label=method_label,
        source_label=source_label,
    )
    archive_path = _zip_artifacts(result["outdir"])
    return hist, benchmark_df, summary, archive_path


with gr.Blocks(title="FD-CATE Demo") as demo:
    gr.Markdown(
        """
        # FD-CATE Demo

        [Paper](https://arxiv.org/abs/2509.22531) | [GitHub](https://github.com/yonghanjung/FD-CATE) | [PyPI](https://pypi.org/project/fd-cate/)

        Estimate heterogeneous treatment effects under front-door identification even when treatment and outcome share hidden confounders.
        Run the canonical synthetic example or upload your own CSV with binary treatment and mediator columns.
        A common uploaded schema is `outcome=y`, `treatment=t`, `mediator=m`, with numeric covariates in the remaining columns.
        """
    )
    data_source = gr.Radio(
        label="Data source",
        choices=["Canonical example", "Upload CSV"],
        value="Canonical example",
    )
    csv_file = gr.File(label="CSV upload", type="filepath")
    with gr.Row():
        outcome = gr.Textbox(label="Outcome column", value="y")
        treatment = gr.Textbox(label="Treatment column", value="t")
        mediator = gr.Textbox(label="Mediator column", value="m")
    covariates = gr.Textbox(
        label="Covariate columns (comma-separated; blank = canonical columns or inferred upload columns)",
        value="",
    )
    with gr.Row():
        n = gr.Slider(label="Canonical example sample size (n)", minimum=100, maximum=600, step=50, value=300)
        d = gr.Slider(label="Canonical example feature dimension (d)", minimum=2, maximum=10, step=1, value=6)
        seed = gr.Number(label="Seed", value=2026, precision=0)
    with gr.Row():
        method = gr.Radio(label="Method", choices=["FD-PI", "FD-DR", "FD-R"], value="FD-DR")
        nuisance = gr.Radio(label="Nuisance learner", choices=["xgb", "nn"], value="xgb")
        include_benchmark = gr.Checkbox(label="Include quick benchmark", value=True)

    run_btn = gr.Button("Run demo", variant="primary")

    plot = gr.Plot(label="Estimated effect distribution")
    bench = gr.Dataframe(label="Quick benchmark RMSE summary", interactive=False)
    summary = gr.Markdown()
    bundle = gr.File(label="Download artifact bundle")

    run_btn.click(
        fn=run_space_demo,
        inputs=[
            data_source,
            csv_file,
            outcome,
            treatment,
            mediator,
            covariates,
            n,
            d,
            method,
            nuisance,
            seed,
            include_benchmark,
        ],
        outputs=[plot, bench, summary, bundle],
    )


if __name__ == "__main__":
    demo.launch()
