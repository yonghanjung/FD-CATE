from __future__ import annotations

import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPACE_APP = ROOT / "scripts" / "hf_space_app.py"
SPACE_README = ROOT / "scripts" / "hf_space_README.md"


def _load_space_namespace():
    pytest.importorskip("gradio")
    return runpy.run_path(str(SPACE_APP), run_name="__space_test__")


def test_hf_space_readme_pin_contract():
    text = SPACE_README.read_text(encoding="utf-8")
    assert 'python_version: "3.10"' in text
    assert 'sdk_version: "5.50.0"' in text


def test_hf_space_api_info_smoke():
    namespace = _load_space_namespace()
    demo = namespace["demo"]
    assert demo.get_api_info()


def test_hf_space_callback_smoke():
    namespace = _load_space_namespace()
    run_space_demo = namespace["run_space_demo"]

    plot, benchmark_df, summary, archive_path = run_space_demo(
        n=100,
        d=4,
        method_label="FD-DR",
        nuisance_learner="xgb",
        seed=0,
        include_benchmark=False,
    )

    assert plot is not None
    assert benchmark_df is not None
    assert "Estimated ATE" in summary
    assert Path(archive_path).exists()
