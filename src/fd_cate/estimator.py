"""sklearn-style estimator wrapper around the legacy FD-CATE implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

from ._version import __version__
from .diagnostics import compute_diagnostics
from .io import _as_1d_float, _as_2d_float, _ensure_binary


@dataclass
class ModelMeta:
    model_format_version: str
    package_version: str
    compatibility_policy: str


def _major_minor(ver: str) -> str:
    parts = ver.split(".")
    if len(parts) < 2:
        return ver
    return f"{parts[0]}.{parts[1]}"


class FDCATE(BaseEstimator):
    """Front-door CATE estimator wrapper with paper-parity defaults."""

    def __init__(
        self,
        *,
        method: str = "fd-dr",
        nuisance_learner: str = "xgb",
        cv: str = "auto",
        random_state: int = 0,
        n_jobs: int = -1,
        inference: str = "none",
        n_bootstrap: int = 200,
        verbose: int = 0,
    ) -> None:
        self.method = method
        self.nuisance_learner = nuisance_learner
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.inference = inference
        self.n_bootstrap = n_bootstrap
        self.verbose = verbose

    def _validate_config(self) -> None:
        if self.method not in {"fd-dr", "fd-r", "fd-pi"}:
            raise ValueError("method must be one of {'fd-dr','fd-r','fd-pi'}")
        if self.nuisance_learner not in {"xgb", "nn"}:
            raise ValueError("nuisance_learner must be one of {'xgb','nn'}")
        if self.inference not in {"none", "bootstrap"}:
            raise ValueError("inference must be 'none' or 'bootstrap'")

    def fit(self, X: Any, y: Any, *, t: Any, m: Any) -> "FDCATE":
        """Fit the estimator using keyword-only treatment/mediator arrays."""
        self._validate_config()

        C = _as_2d_float("X", X)
        Y = _as_1d_float("y", y)
        T = _ensure_binary("t", _as_1d_float("t", t))
        M = _ensure_binary("m", _as_1d_float("m", m))

        if not (len(C) == len(Y) == len(T) == len(M)):
            raise ValueError("X, y, t, and m must have the same number of rows.")

        from FDCATE import (
            fit_folds,
            tau_fd_dr_oof,
            tau_fd_r_3way_oof_smoothed,
            tau_naive_oof,
        )

        bounds_y = None
        bounds_z = None
        folds = fit_folds(C, T, M, Y, self.random_state, learner=self.nuisance_learner)

        if self.method == "fd-pi":
            tau_train = tau_naive_oof(C, T, M, Y, folds, 0.0, bounds_y, bounds_z, self.random_state)
        elif self.method == "fd-dr":
            tau_train, _ = tau_fd_dr_oof(C, T, M, Y, folds, 0.0, bounds_y, bounds_z, self.random_state)
        else:
            tau_train, _ = tau_fd_r_3way_oof_smoothed(
                C,
                T,
                M,
                Y,
                0.0,
                bounds_y,
                bounds_z,
                self.random_state,
                nuisance_learner=self.nuisance_learner,
            )

        tau_train = np.asarray(tau_train, dtype=float)
        finite_mask = np.isfinite(tau_train)
        if not np.any(finite_mask):
            raise RuntimeError("All estimated CATE values are non-finite.")

        self._effect_model = Ridge(alpha=1e-6)
        self._effect_model.fit(C[finite_mask], tau_train[finite_mask])

        self.tau_train_ = tau_train
        self.ate_ = float(np.nanmean(tau_train))
        self.diagnostics_ = compute_diagnostics(
            C,
            Y,
            T,
            M,
            random_state=self.random_state,
        )
        self.n_features_in_ = int(C.shape[1])
        self.feature_names_in_ = None
        self._fitted = True
        return self

    def effect(self, X_new: Any, *, t0: float = 0, t1: float = 1) -> np.ndarray:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Call fit() before effect().")
        X_arr = _as_2d_float("X_new", X_new)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X_new has {X_arr.shape[1]} features but model expects {self.n_features_in_}."
            )
        scale = float(t1) - float(t0)
        return self._effect_model.predict(X_arr) * scale

    def summary(self) -> str:
        if not getattr(self, "_fitted", False):
            return "FDCATE(not fitted)"
        overlap = self.diagnostics_["overlap_proxy"]
        first_stage = self.diagnostics_["first_stage_strength"]["mean_m_t1_minus_t0"]
        warns = self.diagnostics_.get("warnings", [])
        warns_txt = "None" if not warns else "; ".join(warns)
        return (
            "FD-CATE Summary\n"
            f"method={self.method}, nuisance_learner={self.nuisance_learner}, cv={self.cv}\n"
            f"ate={self.ate_:.6f}\n"
            f"overlap_extreme_rates: p_t={overlap['p_t_extreme_rate']:.4f}, p_m={overlap['p_m_extreme_rate']:.4f}\n"
            f"first_stage: E[M|T=1]-E[M|T=0]={first_stage:.6f}\n"
            f"warnings={warns_txt}\n"
            "Interpretation warning: causal interpretation requires front-door assumptions."
        )

    def save(self, path: str | Path) -> None:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Call fit() before save().")
        payload = {
            "meta": asdict(
                ModelMeta(
                    model_format_version="0.1",
                    package_version=__version__,
                    compatibility_policy="Load allowed only when package major.minor matches saved model.",
                )
            ),
            "estimator": self,
        }
        joblib.dump(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "FDCATE":
        payload = joblib.load(Path(path))
        meta = payload.get("meta", {})
        saved = str(meta.get("package_version", "0.0.0"))
        current = __version__
        if _major_minor(saved) != _major_minor(current):
            raise RuntimeError(
                "Incompatible model package version: "
                f"saved={saved}, current={current}. "
                "Compatibility policy: same major.minor only."
            )
        est = payload.get("estimator")
        if not isinstance(est, cls):
            raise RuntimeError("Model payload did not contain a valid FDCATE estimator.")
        return est
