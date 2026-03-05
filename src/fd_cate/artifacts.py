"""Artifact contract writers for fd-cate v0."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ._version import __version__


def _now_iso_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def build_results_json(
    *,
    estimator: Any,
    input_spec: Dict[str, Any],
    command_line: Optional[str],
    data_hash: Optional[str],
) -> Dict[str, Any]:
    warnings = list(estimator.diagnostics_.get("warnings", []))
    return {
        "schema_name": "fdcate.results",
        "schema_version": 0,
        "package_version": __version__,
        "provenance": {
            "timestamp": _now_iso_local(),
            "git_commit": None,
            "random_seed": int(estimator.random_state),
            "command_line": command_line,
            "data_hash": data_hash,
        },
        "assumptions": {
            "frontdoor": [
                "no unmeasured confounding on T->M given X",
                "no direct effect T->Y except through M",
                "no unmeasured confounding on M->Y given (T,X)",
            ],
            "overlap": ["positivity for P(T|X), P(M|T,X)"],
        },
        "input_spec": input_spec,
        "estimator": {
            "method": estimator.method,
            "nuisance_learner": estimator.nuisance_learner,
            "fd_r_b_learner": getattr(estimator, "fd_r_b_learner", "xgb"),
            "fd_r_g_solver": getattr(estimator, "fd_r_g_solver", "direct"),
            "fd_r_swap_average": bool(getattr(estimator, "fd_r_swap_average", True)),
            "cv": estimator.cv,
        },
        "outputs": {
            "ate": float(estimator.ate_),
            "cate": {"saved_as": "effects.csv"},
        },
        "diagnostics": estimator.diagnostics_,
        "warnings": warnings,
    }


def write_artifacts(
    *,
    outdir: str | Path,
    estimator: Any,
    effects: np.ndarray,
    input_spec: Dict[str, Any],
    command_line: Optional[str] = None,
    data_hash: Optional[str] = None,
) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    effects_path = out / "effects.csv"
    pd.DataFrame({"tau": np.asarray(effects).reshape(-1)}).to_csv(effects_path, index=False)

    results = build_results_json(
        estimator=estimator,
        input_spec=input_spec,
        command_line=command_line,
        data_hash=data_hash,
    )

    results_path = out / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=_json_default) + "\n", encoding="utf-8")

    diagnostics_path = out / "diagnostics.json"
    diagnostics_payload = {
        "schema_name": "fdcate.diagnostics",
        "schema_version": 0,
        "package_version": __version__,
        "provenance": {
            "timestamp": _now_iso_local(),
            "random_seed": int(estimator.random_state),
        },
        "diagnostics": estimator.diagnostics_,
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics_payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    summary_path = out / "summary.txt"
    summary_path.write_text(estimator.summary() + "\n", encoding="utf-8")

    return {
        "effects": str(effects_path),
        "results": str(results_path),
        "diagnostics": str(diagnostics_path),
        "summary": str(summary_path),
    }
