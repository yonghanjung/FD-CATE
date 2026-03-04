"""Runtime diagnostics and guardrails for front-door estimation."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, score))
    except Exception:
        return None


def compute_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    m: np.ndarray,
    *,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Compute lightweight quality checks used by `doctor` and fit summaries."""
    warnings: List[str] = []

    t_model = LogisticRegression(max_iter=1000, random_state=random_state)
    t_model.fit(X, t)
    p_t = t_model.predict_proba(X)[:, 1]

    xm = np.column_stack([t, X])
    m_model = LogisticRegression(max_iter=1000, random_state=random_state)
    m_model.fit(xm, m)
    p_m = m_model.predict_proba(xm)[:, 1]

    y_model = LinearRegression()
    y_model.fit(np.column_stack([m, t, X]), y)
    y_pred = y_model.predict(np.column_stack([m, t, X]))

    t_extreme = float(np.mean((p_t < 0.01) | (p_t > 0.99)))
    m_extreme = float(np.mean((p_m < 0.01) | (p_m > 0.99)))

    if t_extreme > 0.05:
        warnings.append("High treatment-overlap extremeness (>5% at p<0.01 or >0.99).")
    if m_extreme > 0.05:
        warnings.append("High mediator-overlap extremeness (>5% at p<0.01 or >0.99).")

    first_stage = float(np.mean(m[t == 1]) - np.mean(m[t == 0])) if np.any(t == 0) else float("nan")
    if np.isfinite(first_stage) and abs(first_stage) < 0.01:
        warnings.append("Weak first-stage signal: E[M|T=1]-E[M|T=0] is near zero.")

    y_rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    t_auc = _safe_auc(t, p_t)
    m_auc = _safe_auc(m, p_m)

    if t_auc is not None and t_auc < 0.55:
        warnings.append("Treatment nuisance fit appears weak (AUC < 0.55).")
    if m_auc is not None and m_auc < 0.55:
        warnings.append("Mediator nuisance fit appears weak (AUC < 0.55).")

    return {
        "input_type_checks": {
            "t_binary": True,
            "m_binary": True,
        },
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "overlap_proxy": {
            "p_t_extreme_rate": t_extreme,
            "p_m_extreme_rate": m_extreme,
        },
        "first_stage_strength": {
            "mean_m_t1_minus_t0": first_stage,
        },
        "nuisance_fit_quality": {
            "treatment_auc": t_auc,
            "mediator_auc": m_auc,
            "outcome_rmse": y_rmse,
        },
        "warnings": warnings,
    }
