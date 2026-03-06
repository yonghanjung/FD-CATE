from __future__ import annotations

import runpy
from pathlib import Path

import pandas as pd
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
    assert "https://arxiv.org/abs/2509.22531" in text
    assert "https://github.com/yonghanjung/FD-CATE" in text


def test_hf_space_api_info_smoke():
    namespace = _load_space_namespace()
    demo = namespace["demo"]
    assert demo.get_api_info()


def test_hf_space_callback_smoke():
    namespace = _load_space_namespace()
    run_space_demo = namespace["run_space_demo"]

    plot, benchmark_df, summary, archive_path = run_space_demo(
        data_source="Canonical example",
        csv_path=None,
        outcome_col="y",
        treatment_col="t",
        mediator_col="m",
        covariates_text="x0,x1,x2,x3",
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


def test_hf_space_callback_smoke_with_uploaded_csv(tmp_path):
    namespace = _load_space_namespace()
    run_space_demo = namespace["run_space_demo"]

    csv_path = tmp_path / "uploaded.csv"
    pd.DataFrame(
        {
            "y": [0.1, 0.4, 0.2, 0.6, 0.8, 0.5],
            "t": [0, 1, 0, 1, 1, 0],
            "m": [0, 1, 0, 1, 1, 0],
            "x0": [1.0, 0.3, 0.1, 0.7, 0.9, 0.2],
            "x1": [0.2, 0.5, 0.7, 0.4, 0.8, 0.1],
        }
    ).to_csv(csv_path, index=False)

    plot, benchmark_df, summary, archive_path = run_space_demo(
        data_source="Upload CSV",
        csv_path=str(csv_path),
        outcome_col="y",
        treatment_col="t",
        mediator_col="m",
        covariates_text="x0,x1",
        n=100,
        d=4,
        method_label="FD-PI",
        nuisance_learner="xgb",
        seed=0,
        include_benchmark=False,
    )

    assert plot is not None
    assert benchmark_df is not None
    assert "uploaded CSV" in summary
    assert Path(archive_path).exists()
