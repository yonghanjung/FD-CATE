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


def test_cli_help_runs():
    result = _run_cli(["--help"])
    assert result.returncode == 0
    assert "fdcate" in result.stdout
    assert "demo" in result.stdout


def test_help_shows_demo_defaults_doc_hint():
    result = _run_cli(["demo", "--help"])
    assert result.returncode == 0
    normalized = " ".join(result.stdout.split()).replace("run- benchmark", "run-benchmark")
    assert "Defaults: n=120, d=6, seed=2026, method=fd-dr, nuisance-learner=xgb" in normalized
    assert "run-benchmark=true." in normalized
