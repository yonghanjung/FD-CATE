import json
import os
import subprocess
import sys
from pathlib import Path


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
