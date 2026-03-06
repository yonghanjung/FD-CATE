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

from fd_cate.demo import run_demo


METHOD_MAP = {
    "FD-PI": "fd-pi",
    "FD-DR": "fd-dr",
    "FD-R": "fd-r",
}


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


def _summary_markdown(result: dict, benchmark_table: pd.DataFrame) -> str:
    artifacts = "\n".join(f"- `{Path(path).name}`" for path in result["artifacts"])
    lines = [
        "## Run summary",
        f"- Estimated ATE: `{result['ate']:.6f}`",
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


def run_space_demo(
    n: int,
    d: int,
    method_label: str,
    nuisance_learner: str,
    seed: int,
    include_benchmark: bool,
):
    tmpdir = Path(tempfile.mkdtemp(prefix="fdcate-space-"))
    result = run_demo(
        outdir=tmpdir / "fdcate-demo",
        n=int(n),
        d=int(d),
        seed=int(seed),
        method=METHOD_MAP[method_label],
        nuisance_learner=nuisance_learner,
        run_benchmark=bool(include_benchmark),
        fd_r_b_learner=nuisance_learner,
    )

    effects_path = result["fit_out"] / "effects.csv"
    hist = _effects_histogram(effects_path, method_label)
    benchmark_df = _benchmark_table(result["benchmark"])
    summary = _summary_markdown(result, benchmark_df)
    archive_path = _zip_artifacts(result["outdir"])
    return hist, benchmark_df, summary, archive_path


with gr.Blocks(title="FD-CATE Demo") as demo:
    gr.Markdown(
        """
        # FD-CATE Demo

        Personalized causal inference under unmeasured confounding via front-door identification.
        This demo generates a synthetic front-door dataset, fits `FD-PI`, `FD-DR`, or `FD-R`,
        and returns an estimated effect distribution plus a downloadable artifact bundle.
        """
    )
    with gr.Row():
        n = gr.Slider(label="Sample size (n)", minimum=100, maximum=600, step=50, value=300)
        d = gr.Slider(label="Feature dimension (d)", minimum=2, maximum=10, step=1, value=6)
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
        inputs=[n, d, method, nuisance, seed, include_benchmark],
        outputs=[plot, bench, summary, bundle],
    )


if __name__ == "__main__":
    demo.launch()
