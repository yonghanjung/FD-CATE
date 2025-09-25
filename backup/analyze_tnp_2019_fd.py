# analyze_tnp_2019_fd.py
# Default input: data/tnp/iu3g-qa69.csv
# Usage:
#   python analyze_tnp_2019_fd.py
#   python analyze_tnp_2019_fd.py --in data/tnp/iu3g-qa69.csv --outcome asinh_tip
#
# Requires: pandas, numpy, (pyarrow if you read parquet), your local FDCATE.py
# Estimand: ATT under one-sided compliance (Z=1 => X=1).

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---- 1) Utility: map authorization strings to {0,1}
def _to_bool_authorized(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in ("true","t","yes","y","1"):  return 1.0
    if s in ("false","f","no","n","0"):  return 0.0
    return np.nan

# ---- 2) Preprocess: df -> (C,X,Z,Y)  [PATCHED]
def preprocess_for_fdcate(df: pd.DataFrame, outcome: str = "asinh_tip_over_fare",
                          topK_area: int = 40, one_hot: bool = False):
    import numpy as np
    import pandas as pd

    # Coerce core types
    df["shared_trip_authorized_bool"] = df["shared_trip_authorized"].map(_to_bool_authorized).astype("float")
    for c in ["trips_pooled","tip","fare","additional_charges","trip_total","trip_miles","trip_seconds"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trip_start_timestamp"] = pd.to_datetime(df["trip_start_timestamp"], errors="coerce")

    # Treatment X and mediator Z
    X = (df["shared_trip_authorized_bool"] == 1.0).astype(int).to_numpy()
    Z = (df["trips_pooled"] >= 2).astype(int).to_numpy()

    # Outcome Y
    if outcome == "tipped_binary":
        Y = (df["tip"] > 0).astype(int).to_numpy()
    elif outcome == "asinh_tip":
        Y = np.arcsinh(df["tip"].fillna(0.0)).to_numpy()
    elif outcome == "asinh_tip_over_fare":
        denom = df["fare"].replace(0, np.nan)
        Y = np.arcsinh((df["tip"]/denom).replace([np.inf,-np.inf], np.nan).fillna(0.0)).to_numpy()
    else:
        raise ValueError("unknown outcome option")

    # Full fare (fare + additional charges), fallback with trip_total - tip if needed
    full_fare = df["fare"].fillna(0) + df["additional_charges"].fillna(0)
    missing_full = full_fare.isna() | (full_fare == 0)
    if missing_full.any():
        backup = (df["trip_total"] - df["tip"]).where(~df["trip_total"].isna() & ~df["tip"].isna())
        full_fare = full_fare.where(~missing_full, backup)

    # Time features (use lowercase 'h' to avoid FutureWarning)
    ts = df["trip_start_timestamp"]
    dow = ts.dt.dayofweek.astype("Int8")   # 0..6
    hod = ts.dt.hour.astype("Int8")        # 0..23
    twohr = (ts.dt.floor("2h") - ts.min()).dt.total_seconds() // 7200  # integer bins rel. to min time

    # Geography; bucket rare areas into -1
    for ca in ["pickup_community_area","dropoff_community_area"]:
        df[ca] = pd.to_numeric(df[ca], errors="coerce").astype("Int16")
        topK = df[ca].value_counts(dropna=True).index[:topK_area]
        df[ca] = df[ca].where(df[ca].isin(topK), -1)

    C_df = pd.DataFrame({
        "full_fare": full_fare,
        "log_miles": np.log1p(df["trip_miles"].clip(lower=0)),
        "log_secs":  np.log1p(df["trip_seconds"].clip(lower=0)),
        "twohr":     twohr.astype("Int32"),
        "dow":       dow,
        "hod":       hod,
        "pca":       df["pickup_community_area"].astype("Int16"),
        "dca":       df["dropoff_community_area"].astype("Int16"),
    })

    if one_hot:
        C = pd.get_dummies(C_df, columns=["dow","hod","pca","dca"], dummy_na=False)
    else:
        C = C_df

    # >>> KEY FIX: force a pure numeric matrix before isfinite <<<
    # (pandas extension ints -> float64; booleans/uint8 also -> float64)
    C = C.apply(pd.to_numeric, errors="coerce")
    C_np = C.astype("float64").to_numpy()

    # Also ensure these are numeric arrays (float ok for isfinite)
    X_np = X.astype("float64")
    Z_np = Z.astype("float64")
    Y_np = Y.astype("float64")

    # Finite mask
    M = np.isfinite(C_np).all(axis=1) & np.isfinite(X_np) & np.isfinite(Z_np) & np.isfinite(Y_np)

    # Filter and set final dtypes expected by your learners
    C_np = C_np[M, :]
    X_np = X[M].astype(int)
    Z_np = Z[M].astype(int)
    Y_np = Y[M].astype(float)

    return C_np, X_np, Z_np, Y_np, M, C.columns.tolist()


# ---- 3) Small diagnostics  [PATCHED minor robustness]
def quick_diag(X, Z, Y):
    import numpy as np
    n = len(Y)
    print(f"[diag] n={n:,}")
    # Robust binary check for Y
    y_nonan = Y[np.isfinite(Y)]
    uniq = np.unique(y_nonan)
    is_binary_y = (len(uniq) <= 2) and np.all(np.isin(uniq, [0, 1]))
    y1 = (y_nonan.mean() if is_binary_y else float("nan"))
    print(f"[diag] P(X=1)={X.mean():.3f}, P(Z=1)={Z.mean():.3f}, P(Y=1)~{y1:.3f}")
    bad = int(np.sum((Z==1) & (X==0)))  # one-sided compliance violations
    print(f"[diag] one-sided compliance violations (Z=1 & X=0): {bad}")


# ---- 4) Wire up to your FDCATE.py
def run_fd_estimators(C, X, Z, Y, seed=2025, delta=0.0):
    from FDCATE import fit_folds, tau_fd_dr_oof, tau_fd_r_3way_oof_smoothed, tau_naive_oof
    bound_z = [0,1]
    bound_y = [0,1]
    folds = fit_folds(C, X, Z, Y, seed)
    tau_dr, tau_dr_model = tau_fd_dr_oof(C, X, Z, Y, folds, delta,bound_y,bound_z, seed)
    tau_r, tau_r_model  = tau_fd_r_3way_oof_smoothed(C, X, Z, Y, delta,bound_y,bound_z, seed)
    tau_naive = tau_naive_oof(C, X, Z, Y, folds, delta,bound_y,bound_z, seed)
    return tau_dr, tau_r, tau_naive, tau_dr_model, tau_r_model

def summarize(name, tau):
    finite = np.isfinite(tau)
    v = tau[finite]
    q = lambda p: np.nanpercentile(v, p)
    print(f"{name}: n={finite.sum():,}  ATE/ATT≈{v.mean():.6f} | q10={q(10):.6f}  q50={q(50):.6f}  q90={q(90):.6f}")

def shap_to_2d(V):
    """
    Normalize SHAP output to shape (n_samples, n_features).
    - list of arrays -> average over outputs
    - 3D array (n, p, k) -> average over outputs (axis=-1)
    """
    if isinstance(V, list):
        V = np.mean(np.stack(V, axis=0), axis=0)
    elif getattr(V, "ndim", None) == 3:
        V = V.mean(axis=-1)
    return V


if __name__ == "__main__":
    # ---- 5) CLI (now with a DEFAULT pointing to data/tnp/iu3g-qa69.csv)
    argv = None
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/tnp/iu3g-qa69.csv",
                    help="Path to local iu3g-qa69 slice (default: data/tnp/iu3g-qa69.csv)")
    ap.add_argument("--outcome", default="asinh_tip_over_fare",
                    choices=["tipped_binary","asinh_tip","asinh_tip_over_fare"])
    ap.add_argument("--topK_area", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--no_one_hot", action="store_true")
    args = ap.parse_args(argv)

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"[error] Input file not found: {in_path}")

    print(f"Loading {in_path} …")
    # Parquet vs CSV (gz is fine; pandas infers)
    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path, low_memory=False)

    print("Preprocessing → (C,X,Z,Y)…")
    C, X, Z, Y, mask, c_names = preprocess_for_fdcate(
        df, outcome=args.outcome, topK_area=args.topK_area, one_hot=False
    )

    quick_diag(X, Z, Y)

    print("Running FD-DR, FD-R, FD-naive…")
    tau_dr, tau_r, tau_naive, tau_dr_model, tau_r_model = run_fd_estimators(C, X, Z, Y, seed=args.seed, delta=args.delta)

    # Summaries
    summarize("FD-DR", tau_dr)
    summarize("FD-R ", tau_r)
    summarize("FD-naive", tau_naive)
    naive = Y[X==1].mean() - Y[X==0].mean()
    print(f"Naive diff-in-means (E[Y|X=1]-E[Y|X=0]): {naive:.6f}")

    # Save for later plots/tables, next to the input file
    out_npz = in_path.with_suffix("").as_posix() + f".{args.outcome}.fdcate.npz"
    np.savez(out_npz, tau_dr=tau_dr, tau_r=tau_r, tau_naive = tau_naive, X=X, Z=Z, Y=Y, C_shape=C.shape)
    print(f"Saved results to {out_npz}")
    
    import shap
    import matplotlib.pyplot as plt
    
    shap_model_dr = shap.LinearExplainer(tau_dr_model, shap.maskers.Independent(C))
    shap_model_r = shap.LinearExplainer(tau_r_model, shap.maskers.Independent(C))
    
    shap_value_dr = shap_model_dr.shap_values(C)  # C as DataFrame
    shap_value_r = shap_model_r.shap_values(C)  # C as DataFrame
    
    feature_map = {
        1: "Fare",
        2: "Miles",
        3: "Sec",
        4: "TwoHr",
        5: "dow",
        6: "hod",
        7: "pca",
        8: "dca",
    }
    
    # Ensure C is a DataFrame
    if not isinstance(C, pd.DataFrame):
        C = pd.DataFrame(C)

    # Assign columns as 1, 2, 3, ...
    # {1: "full_fare", 2: "log_miles", 3: "log_secs", 4:"twohr", 5:"dow", 6: "hod", 7: "pca" and 8:"dca"}
    C.columns = range(1, C.shape[1] + 1)
    
    # Apply the requested names (keeps any unmapped columns as their numeric labels)
    C.columns = [feature_map.get(int(col), str(col)) for col in C.columns]
    
    # Convert to 2D
    V_dr = shap_to_2d(shap_value_dr)
    V_r  = shap_to_2d(shap_value_r)

    # --- 2) Mean |SHAP| per feature ---
    imp_dr_df = (
        pd.DataFrame({"feature": C.columns, "mean_abs_shap_dr": np.abs(V_dr).mean(axis=0)})
        .sort_values("mean_abs_shap_dr", ascending=False)
        .reset_index(drop=True)
    )
    imp_r_df = (
        pd.DataFrame({"feature": C.columns, "mean_abs_shap_r": np.abs(V_r).mean(axis=0)})
        .sort_values("mean_abs_shap_r", ascending=False)
        .reset_index(drop=True)
    )
    
    # --- 3) Combine & sort by the larger of the two importances ---
    combined = (
        imp_dr_df.merge(imp_r_df, on="feature", how="outer")
                .fillna(0.0)
    )
    order_key = combined[["mean_abs_shap_dr", "mean_abs_shap_r"]].max(axis=1)
    combined = combined.loc[order_key.sort_values(ascending=False).index].reset_index(drop=True)

    # Optionally, display top-k only
    topk = min(20, combined.shape[0])
    plot_df = combined.head(topk)

    # Assume plot_df has: feature, mean_abs_shap_dr, mean_abs_shap_r
    # Sort by whichever order you like (here, by combined max importance)
    order_key = plot_df[["mean_abs_shap_dr", "mean_abs_shap_r"]].max(axis=1)
    plot_df = plot_df.loc[order_key.sort_values(ascending=True).index]  # ascending so top = biggest

    y = np.arange(len(plot_df))

    plt.figure(figsize=(8, 6))

    # DR bars go left (negative)
    plt.barh(y, -plot_df["mean_abs_shap_dr"], color="#377EB8", label="FD-DR")

    # R bars go right (positive)
    plt.barh(y,  plot_df["mean_abs_shap_r"], color="#4DAF4A", label="FD-R")

    # Feature labels in the middle
    

    # Axis formatting
    plt.axvline(0, color="k", linewidth=0.8)
    # plt.xlabel("Mean |SHAP|")
    plt.xticks(fontsize=20)
    # plt.yticks([])
    plt.yticks(y, plot_df["feature"].astype(str))
    # plt.title("Global SHAP importance — FD-DR (left) vs FD-R (right)")
    # plt.legend(loc="upper right")
    plt.tight_layout()
    
    out_dir_fig = "simulation/plot/"
    VERSION_SAVE = "260923_1400_final"
    plt.savefig(out_dir_fig + "SHAP_analysis" + f"_{VERSION_SAVE}.pdf")
    plt.show()
    
    
    # --- STEP 1: overlay histogram for tau_r and tau_dr (one figure) ---
    # (requires: import matplotlib.pyplot as plt at the top of your file)
    xrange = (-0.15, 0.05)
    # Keep only finite values and flatten
    mask = np.isfinite(tau_dr) & np.isfinite(tau_r)
    _tau_dr = tau_dr[mask].reshape(-1)
    _tau_r  = tau_r[mask].reshape(-1)

    # Common histogram bins and range so the two are directly comparable
    xmin = np.min(np.concatenate([_tau_dr, _tau_r]))
    xmax = np.max(np.concatenate([_tau_dr, _tau_r]))
    bins = 80  # keep it simple; you can adjust to taste

    # Where to save (same folder as your input file)
    hist_path = out_dir_fig + "histogram" + f"_{VERSION_SAVE}.pdf"

    plt.figure(figsize=(8, 6))
    plt.hist(_tau_dr, bins=bins,  alpha=0.55, range=xrange,
         label="FD-DR", histtype="stepfilled", linewidth=1.0, color="#377EB8")
    plt.hist(_tau_r,  bins=bins, alpha=0.45, range=xrange,
         label="FD-R",  histtype="stepfilled", linewidth=1.0, color="#4DAF4A")

    # Optional: vertical mean lines (nice but still simple)
    # plt.axvline(_tau_dr.mean(), linestyle="--", linewidth=2.0)
    # plt.axvline(_tau_r.mean(),  linestyle="--", linewidth=2.0)

    # plt.title("Distribution of $\\hat{\\tau}$: FD-DR vs FD-R")
    # plt.xlabel(r"$\hat{\tau}$")
    plt.xticks(fontsize=15)
    
    # --- Custom y-axis ticks ---
    import matplotlib.ticker as mticker
    ax = plt.gca()
    # Set tick positions (units are raw counts)
    ax.set_yticks([200_000, 400_000, 600_000, 800_000])
    # Format them as "200k", "400k", …
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    
    plt.yticks(fontsize=20)
    # plt.ylabel("count")
    # plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(hist_path, dpi=160, bbox_inches="tight")
    print(f"[fig] saved histogram overlay → {hist_path}")
    plt.show()
    plt.close()

    
    
    
    # --- STEP 2: policy value plot (mean tau among top-alpha) ---
    def _pvc_curve(tauhat, alphas=None):
        """Return (alphas, mean of top-alpha tauhat)."""
        if alphas is None:
            alphas = np.linspace(0.01, 1.0, 200)
        v = np.asarray(tauhat).reshape(-1)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return alphas, np.full_like(alphas, np.nan, dtype=float)
        v = np.sort(v)[::-1]  # largest first
        n = v.size
        means = []
        for a in alphas:
            k = max(1, int(np.floor(a * n)))
            means.append(np.mean(v[:k]))
        return alphas, np.array(means)

    # Build curves
    curves = []
    alphas = np.linspace(0.01, 1.0, 200)

    a_dr, m_dr = _pvc_curve(tau_dr, alphas)
    curves.append(("FD-DR", a_dr, m_dr, "#377EB8"))

    a_r, m_r = _pvc_curve(tau_r, alphas)
    curves.append(("FD-R", a_r, m_r, "#4DAF4A"))

    # Save path (same base as your input file)
    pvc_path = out_dir_fig + "policy_plot" + f"_{VERSION_SAVE}.pdf"

    # Plot
    plt.figure(figsize=(8, 6))
    for name, a, m, color in curves:
        plt.plot(a, m, label=name, linewidth=2.0, color = color)
        # Mark endpoints (nice for reading)
        plt.scatter([a[0], a[-1]], [m[0], m[-1]], s=25, color=color)
        # plt.text(a[0], m[0], f"{name} @ {a[0]:.2f}: {m[0]:.4f}", va="bottom", ha="left", fontsize=10)
        # plt.text(a[-1], m[-1], f"{name} @ 1.00: {m[-1]:.4f}",   va="top",    ha="right", fontsize=10)

    # plt.title("Policy Value Curve: mean $\\hat{\\tau}$ among top-$\\alpha$")
    # plt.xlabel(r"$\alpha$ (fraction targeted by largest $\hat{\tau}$)")
    plt.xticks(fontsize=20)
    # plt.ylabel(r"mean $\hat{\tau}$ in top-$\alpha$")
    plt.yticks(fontsize=20)
    # plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(pvc_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[fig] saved policy value plot → {pvc_path}")


    
