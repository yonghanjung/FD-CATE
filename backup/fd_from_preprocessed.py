
# fd_from_preprocessed.py
# Compute SHAP/permutation importances and group-specific front-door effects
# from preprocessed (C, Z, X, Y).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HAVE_SHAP = True
try:
    import shap
except Exception:
    HAVE_SHAP = False

try:
    import scipy.sparse as sp
    HAVE_SPARSE = True
except Exception:
    HAVE_SPARSE = False

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error

def _to_dataframe(C, feature_names=None):
    # Coerce C to DataFrame with names
    if isinstance(C, pd.DataFrame):
        return C
    if HAVE_SPARSE and getattr(sp, "isspmatrix", lambda x: False)(C):
        C = C.tocsr()
        if feature_names is None:
            feature_names = [f"c{j}" for j in range(C.shape[1])]
        if C.shape[1] <= 2000:
            return pd.DataFrame(C.toarray(), columns=list(feature_names))
        raise RuntimeError("C is sparse with very high dimensionality. Please provide a dense DataFrame subset or feature selection.")
    if isinstance(C, np.ndarray):
        if C.ndim != 2:
            raise ValueError("C must be 2D.")
        if feature_names is None:
            feature_names = [f"c{j}" for j in range(C.shape[1])]
        return pd.DataFrame(C, columns=list(feature_names))
    raise TypeError("Unsupported C type. Use pandas DataFrame, numpy 2D array, or scipy sparse matrix.")

def _subsample_idx(n, k, random_state=123):
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return rng.choice(n, size=k, replace=False)

def _fit_models(C_df, X, Z, Y, outcome_type=None, random_state=123, n_fit=200000):
    n = len(Y)
    idx = _subsample_idx(n, n_fit, random_state)

    C_fit = C_df.iloc[idx]
    X_fit = np.asarray(X)[idx]
    Z_fit = np.asarray(Z)[idx]
    Y_fit = np.asarray(Y)[idx]

    # Stage 1: Z ~ X + C
    C1 = C_fit.assign(__X__=X_fit)
    clf1 = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=10, n_jobs=-1, random_state=random_state)
    X_tr, X_te, y_tr, y_te = train_test_split(C1, Z_fit, test_size=0.2, random_state=random_state)
    clf1.fit(X_tr, y_tr)
    try:
        auc1 = float(roc_auc_score(y_te, clf1.predict_proba(X_te)[:,1]))
    except Exception:
        auc1 = None

    # Stage 2: Y ~ Z + X + C
    C2 = C_fit.assign(__X__=X_fit, __Z__=Z_fit)
    if outcome_type is None:
        outcome_type = "binary" if set(np.unique(Y_fit)).issubset({0,1}) else "continuous"
    if outcome_type == "binary":
        clf2 = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=10, n_jobs=-1, random_state=random_state)
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(C2, Y_fit, test_size=0.2, random_state=random_state)
        clf2.fit(X_tr2, y_tr2)
        try:
            metric2 = float(roc_auc_score(y_te2, clf2.predict_proba(X_te2)[:,1]))
        except Exception:
            metric2 = None
    else:
        clf2 = RandomForestRegressor(n_estimators=600, max_depth=None, min_samples_leaf=10, n_jobs=-1, random_state=random_state)
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(C2, Y_fit, test_size=0.2, random_state=random_state)
        clf2.fit(X_tr2, y_tr2)
        try:
            metric2 = float(np.sqrt(mean_squared_error(y_te2, clf2.predict(X_te2))))
        except Exception:
            metric2 = None

    return {"clf1": clf1, "clf2": clf2, "auc1": auc1, "metric2": metric2, "fit_idx": idx}

def _importance(model, X_df, y, title, out_csv, out_png=None, max_display=40):
    if HAVE_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_df)
            if isinstance(sv, list):
                vals = np.abs(sv[-1]).mean(axis=0)
            else:
                vals = np.abs(sv.values).mean(axis=0)
            imp = pd.DataFrame({"feature": X_df.columns, "mean_abs_shap": vals}).sort_values("mean_abs_shap", ascending=False)
            imp.to_csv(out_csv, index=False)
            if out_png is not None:
                topk = imp.head(max_display).iloc[::-1]
                plt.figure()
                plt.barh(topk["feature"], topk["mean_abs_shap"])
                plt.title(title)
                plt.tight_layout()
                plt.savefig(out_png, dpi=150)
                plt.close()
            return imp
        except Exception:
            pass
    # fallback: permutation
    r = permutation_importance(model, X_df, y, n_repeats=5, n_jobs=-1, random_state=123)
    imp = pd.DataFrame({"feature": X_df.columns, "mean_importance": r.importances_mean, "std": r.importances_std})
    imp = imp.sort_values("mean_importance", ascending=False)
    imp.to_csv(out_csv, index=False)
    if out_png is not None:
        topk = imp.head(max_display).iloc[::-1]
        plt.figure()
        plt.barh(topk["feature"], topk.get("mean_abs_shap", topk["mean_importance"]))
        plt.title(title + " (permutation)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
    return imp

def _gamma_delta_by_group(clf1, clf2, C_df, X, eval_idx, groups, min_group_n):
    rows = []
    C_eval = C_df.iloc[eval_idx]
    X_eval = np.asarray(X)[eval_idx]
    for gname, gvals in groups.items():
        gser = pd.Series(gvals).reindex(C_df.index).iloc[eval_idx]
        for lv in pd.unique(gser.dropna()):
            mask = (gser == lv).to_numpy()
            n = int(mask.sum())
            if n < min_group_n:
                continue
            # gamma_g
            D1 = C_eval.loc[mask].assign(__X__=1)
            D0 = C_eval.loc[mask].assign(__X__=0)
            p1 = clf1.predict_proba(D1)[:,1]
            p0 = clf1.predict_proba(D0)[:,1]
            gamma_g = float(np.mean(p1 - p0))
            # delta_g
            E1 = C_eval.loc[mask].assign(__X__=1, __Z__=1)
            E0 = C_eval.loc[mask].assign(__X__=1, __Z__=0)
            if hasattr(clf2, "predict_proba"):
                y1 = clf2.predict_proba(E1)[:,1]
                y0 = clf2.predict_proba(E0)[:,1]
            else:
                y1 = clf2.predict(E1); y0 = clf2.predict(E0)
            delta_g = float(np.mean(y1 - y0))
            rows.append({"group_var": gname, "level": str(lv), "n": n,
                         "gamma": gamma_g, "delta": delta_g, "tau": gamma_g*delta_g})
    if not rows:
        return pd.DataFrame(columns=["group_var","level","n","gamma","delta","tau"])
    return pd.DataFrame(rows).sort_values("tau")

def run_from_arrays(C, X, Z, Y, groups=None, feature_names=None, outdir=".", random_state=123,
                    outcome_type=None, n_fit=200000, n_eval=200000, min_group_n=500):
    """
    Entry point when you already have preprocessed arrays.
    - C: DataFrame (preferred) or numpy array / scipy.sparse
    - X, Z, Y: 1D arrays (length n)
    - groups: dict of {name: vector of length n} with candidate groupings
    """
    from pathlib import Path
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    C_df = _to_dataframe(C, feature_names=feature_names)
    n = len(Y)
    if len(X)!=n or len(Z)!=n or C_df.shape[0]!=n:
        raise ValueError(f"Length mismatch: C:{C_df.shape[0]}, X:{len(X)}, Z:{len(Z)}, Y:{len(Y)}")

    models = _fit_models(C_df, X, Z, Y, outcome_type=outcome_type, random_state=random_state, n_fit=n_fit)
    clf1, clf2 = models["clf1"], models["clf2"]

    fit_idx = models["fit_idx"]
    imp1 = _importance(clf1, C_df.iloc[fit_idx].assign(__X__=np.asarray(X)[fit_idx]),
                       y=np.asarray(Z)[fit_idx],
                       title="Stage 1: Z~X,C",
                       out_csv=outdir/"stage1_importance.csv",
                       out_png=outdir/"stage1_importance.png")
    imp2 = _importance(clf2, C_df.iloc[fit_idx].assign(__X__=np.asarray(X)[fit_idx], __Z__=np.asarray(Z)[fit_idx]),
                       y=np.asarray(Y)[fit_idx],
                       title="Stage 2: Y~Z,X,C",
                       out_csv=outdir/"stage2_importance.csv",
                       out_png=outdir/"stage2_importance.png")

    eval_idx = _subsample_idx(n, n_eval, random_state+1)
    groups = groups or {}
    ge = _gamma_delta_by_group(clf1, clf2, C_df, X, eval_idx, groups, min_group_n=min_group_n)
    ge_path = outdir/"fd_group_effects.csv"
    ge.to_csv(ge_path, index=False)

    summary = {
        "stage1_importance_csv": str(outdir/"stage1_importance.csv"),
        "stage2_importance_csv": str(outdir/"stage2_importance.csv"),
        "group_effects_csv": str(ge_path),
        "stage1_auc": models["auc1"],
        "stage2_metric": models["metric2"],
        "used_n_for_fit": int(len(fit_idx)),
        "used_n_for_eval": int(len(eval_idx)),
    }
    with open(outdir/"fd_from_preprocessed.summary.json", "w") as f:
        import json; json.dump(summary, f, indent=2)
    print("Wrote summary:", outdir/"fd_from_preprocessed.summary.json")
    return summary
