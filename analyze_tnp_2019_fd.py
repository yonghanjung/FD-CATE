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
def preprocess_for_fdcate(df: pd.DataFrame, outcome: str = "tipped_binary",
                          topK_area: int = 40, one_hot: bool = True):
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
    tau_dr = tau_fd_dr_oof(C, X, Z, Y, folds, delta,bound_y,bound_z, seed)
    tau_r  = tau_fd_r_3way_oof_smoothed(C, X, Z, Y, delta,bound_y,bound_z, seed)
    tau_naive = tau_naive_oof(C, X, Z, Y, folds, delta,bound_y,bound_z, seed)
    return tau_dr, tau_r, tau_naive

def summarize(name, tau):
    finite = np.isfinite(tau)
    v = tau[finite]
    q = lambda p: np.nanpercentile(v, p)
    print(f"{name}: n={finite.sum():,}  ATE/ATT≈{v.mean():.6f} | q10={q(10):.6f}  q50={q(50):.6f}  q90={q(90):.6f}")



if __name__ == "__main__":
    # ---- 5) CLI (now with a DEFAULT pointing to data/tnp/iu3g-qa69.csv)
    argv = None
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/tnp/iu3g-qa69.csv",
                    help="Path to local iu3g-qa69 slice (default: data/tnp/iu3g-qa69.csv)")
    ap.add_argument("--outcome", default="tipped_binary",
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
        df, outcome=args.outcome, topK_area=args.topK_area, one_hot=(not args.no_one_hot)
    )

    quick_diag(X, Z, Y)

    print("Running FD-DR, FD-R, FD-naive…")
    tau_dr, tau_r, tau_naive = run_fd_estimators(C, X, Z, Y, seed=args.seed, delta=args.delta)

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
