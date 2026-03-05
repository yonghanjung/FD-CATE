import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "fd_cate", *args],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def test_demo_creates_expected_artifacts(tmp_path):
    outdir = tmp_path / "demo-out"
    result = _run_cli(
        [
            "demo",
            "--outdir",
            str(outdir),
            "--n",
            "50",
            "--d",
            "4",
            "--seed",
            "2026",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "[demo] next: fdcate effect --model" in result.stdout

    expected = [
        outdir / "synthetic.csv",
        outdir / "fit_out" / "summary.txt",
        outdir / "fit_out" / "results.json",
        outdir / "fit_out" / "diagnostics.json",
        outdir / "fit_out" / "effects.csv",
        outdir / "fit_out" / "model.pkl",
        outdir / "benchmark_quick.json",
    ]
    for path in expected:
        assert path.exists(), f"missing expected artifact: {path}"


def test_demo_defaults_contract(tmp_path):
    outdir = tmp_path / "demo-defaults"
    result = _run_cli(["demo", "--outdir", str(outdir)])
    assert result.returncode == 0, result.stderr
    assert "[demo] output directory:" in result.stdout
    assert "[demo] ATE=" in result.stdout
    assert "[demo] generated files:" in result.stdout
    assert "[demo] next: fdcate effect --model" in result.stdout

    synthetic = outdir / "synthetic.csv"
    assert synthetic.exists()
    df = pd.read_csv(synthetic)
    assert df.shape[0] == 120
    assert set(["y", "t", "m"]).issubset(df.columns)
    assert sum(col.startswith("x") for col in df.columns) == 6

    results = json.loads((outdir / "fit_out" / "results.json").read_text(encoding="utf-8"))
    assert results["estimator"]["method"] == "fd-dr"
    assert results["estimator"]["nuisance_learner"] == "xgb"
    assert (outdir / "benchmark_quick.json").exists()


def test_demo_run_benchmark_false_skips_benchmark_file(tmp_path):
    outdir = tmp_path / "demo-no-bench"
    result = _run_cli(
        [
            "demo",
            "--outdir",
            str(outdir),
            "--n",
            "40",
            "--d",
            "4",
            "--seed",
            "2026",
            "--run-benchmark",
            "false",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert (outdir / "fit_out" / "results.json").exists()
    assert not (outdir / "benchmark_quick.json").exists()


def test_demo_seed_is_deterministic(tmp_path):
    outdir_a = tmp_path / "demo-a"
    outdir_b = tmp_path / "demo-b"
    args = [
        "demo",
        "--n",
        "45",
        "--d",
        "4",
        "--seed",
        "777",
        "--run-benchmark",
        "false",
    ]
    result_a = _run_cli([*args, "--outdir", str(outdir_a)])
    result_b = _run_cli([*args, "--outdir", str(outdir_b)])
    assert result_a.returncode == 0, result_a.stderr
    assert result_b.returncode == 0, result_b.stderr

    payload_a = json.loads((outdir_a / "fit_out" / "results.json").read_text(encoding="utf-8"))
    payload_b = json.loads((outdir_b / "fit_out" / "results.json").read_text(encoding="utf-8"))
    assert payload_a["outputs"]["ate"] == payload_b["outputs"]["ate"]


def test_demo_overwrite_same_outdir_is_deterministic(tmp_path):
    outdir = tmp_path / "demo-overwrite"
    args = [
        "demo",
        "--outdir",
        str(outdir),
        "--n",
        "45",
        "--d",
        "4",
        "--seed",
        "777",
        "--run-benchmark",
        "false",
    ]
    result_a = _run_cli(args)
    assert result_a.returncode == 0, result_a.stderr
    payload_a = json.loads((outdir / "fit_out" / "results.json").read_text(encoding="utf-8"))

    result_b = _run_cli(args)
    assert result_b.returncode == 0, result_b.stderr
    payload_b = json.loads((outdir / "fit_out" / "results.json").read_text(encoding="utf-8"))
    assert payload_a["outputs"]["ate"] == payload_b["outputs"]["ate"]
    assert (outdir / "fit_out" / "model.pkl").exists()


def test_demo_seed_stable_over_five_runs(tmp_path):
    ates = []
    base_args = [
        "demo",
        "--n",
        "40",
        "--d",
        "4",
        "--seed",
        "2026",
        "--run-benchmark",
        "false",
    ]
    for idx in range(5):
        outdir = tmp_path / f"demo-run-{idx}"
        result = _run_cli([*base_args, "--outdir", str(outdir)])
        assert result.returncode == 0, result.stderr
        payload = json.loads((outdir / "fit_out" / "results.json").read_text(encoding="utf-8"))
        ates.append(payload["outputs"]["ate"])

    assert all(ate == ates[0] for ate in ates)


def test_demo_nn_nuisance_smoke(tmp_path):
    outdir = tmp_path / "demo-nn"
    result = _run_cli(
        [
            "demo",
            "--outdir",
            str(outdir),
            "--n",
            "45",
            "--d",
            "4",
            "--seed",
            "21",
            "--nuisance-learner",
            "nn",
            "--run-benchmark",
            "false",
        ]
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((outdir / "fit_out" / "results.json").read_text(encoding="utf-8"))
    assert payload["estimator"]["nuisance_learner"] == "nn"
    assert (outdir / "fit_out" / "effects.csv").exists()


def test_demo_method_fd_r_smoke(tmp_path):
    outdir = tmp_path / "demo-fdr"
    result = _run_cli(
        [
            "demo",
            "--outdir",
            str(outdir),
            "--n",
            "45",
            "--d",
            "4",
            "--seed",
            "22",
            "--method",
            "fd-r",
            "--fd-r-g-solver",
            "ratio",
            "--fd-r-b-learner",
            "nn",
            "--run-benchmark",
            "false",
        ]
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((outdir / "fit_out" / "results.json").read_text(encoding="utf-8"))
    assert payload["estimator"]["method"] == "fd-r"
    assert payload["estimator"]["fd_r_g_solver"] == "ratio"
    assert payload["estimator"]["fd_r_b_learner"] == "nn"


def test_demo_invalid_boolean_arg_fails(tmp_path):
    outdir = tmp_path / "demo-invalid-bool"
    result = _run_cli(
        [
            "demo",
            "--outdir",
            str(outdir),
            "--run-benchmark",
            "maybe",
        ]
    )
    assert result.returncode != 0
    assert "Invalid boolean value" in result.stderr
