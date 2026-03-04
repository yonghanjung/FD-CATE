import os
import subprocess
import sys
from pathlib import Path


def test_cli_help_runs():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-m", "fd_cate", "--help"],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0
    assert "fdcate" in result.stdout
