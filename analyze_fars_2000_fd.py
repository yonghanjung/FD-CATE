# analyze_fars_2000_fd.py
# Default input dir: data/fars/
# Usage:
#   python analyze_fars_2000_fd.py
#   python analyze_fars_2000_fd.py --dir data/fars --seed 2025 --delta 0.0
#
# Requires: pandas, numpy, and your local FDCATE.py providing:
#   from FDCATE import fit_folds, tau_fd_dr_oof, tau_fd_r_3way_oof_smoothed, tau_naive_oof
#
# Estimand in this script:
#   CATE via front-door with X (primary law), Z (belt_used), Y (died), C (unit covariates, ~10-15 dims).
#
# Files expected in --dir:
#   ACCIDENT.CSV, PERSON.CSV, VEHICLE.CSV (optional), MIACC.CSV, MIDRVACC.CSV, MIPER.CSV (unused here),
#   state_primary_2000.csv  (columns: STATE, state_abbr, state_name, primary_2000)
#
# Output:
#   npz with tau_dr, tau_r, tau_naive, plus metadata; a CSV with the cleaned analytic dataset (optional).

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# FARS numeric code -> (USPS, Name)
STATE_CODE_MAP = {
    1:("AL","Alabama"), 2:("AK","Alaska"), 4:("AZ","Arizona"), 5:("AR","Arkansas"),
    6:("CA","California"), 8:("CO","Colorado"), 9:("CT","Connecticut"), 10:("DE","Delaware"),
    11:("DC","District of Columbia"), 12:("FL","Florida"), 13:("GA","Georgia"), 15:("HI","Hawaii"),
    16:("ID","Idaho"), 17:("IL","Illinois"), 18:("IN","Indiana"), 19:("IA","Iowa"),
    20:("KS","Kansas"), 21:("KY","Kentucky"), 22:("LA","Louisiana"), 23:("ME","Maine"),
    24:("MD","Maryland"), 25:("MA","Massachusetts"), 26:("MI","Michigan"), 27:("MN","Minnesota"),
    28:("MS","Mississippi"), 29:("MO","Missouri"), 30:("MT","Montana"), 31:("NE","Nebraska"),
    32:("NV","Nevada"), 33:("NH","New Hampshire"), 34:("NJ","New Jersey"), 35:("NM","New Mexico"),
    36:("NY","New York"), 37:("NC","North Carolina"), 38:("ND","North Dakota"), 39:("OH","Ohio"),
    40:("OK","Oklahoma"), 41:("OR","Oregon"), 42:("PA","Pennsylvania"), 44:("RI","Rhode Island"),
    45:("SC","South Carolina"), 46:("SD","South Dakota"), 47:("TN","Tennessee"), 48:("TX","Texas"),
    49:("UT","Utah"), 50:("VT","Vermont"), 51:("VA","Virginia"), 53:("WA","Washington"),
    54:("WV","West Virginia"), 55:("WI","Wisconsin"), 56:("WY","Wyoming")
}

# Super-regions (roll-up) encoded as a single integer feature:
# East=0, Midwest=1, South=2, West=3
CENSUS_REGION_ID = {
    # Northeast -> East(0)
    "CT":0,"ME":0,"MA":0,"NH":0,"RI":0,"VT":0,"NJ":0,"NY":0,"PA":0,
    # Midwest -> Midwest(1)
    "IL":1,"IN":1,"MI":1,"OH":1,"WI":1,"IA":1,"KS":1,"MN":1,"MO":1,"NE":1,"ND":1,"SD":1,
    # South -> South(2)
    "DE":2,"DC":2,"FL":2,"GA":2,"MD":2,"NC":2,"SC":2,"VA":2,"WV":2,"AL":2,"KY":2,"MS":2,"TN":2,"AR":2,"LA":2,"OK":2,"TX":2,
    # West -> West(3)
    "AZ":3,"CO":3,"ID":3,"MT":3,"NV":3,"NM":3,"UT":3,"WY":3,"AK":3,"CA":3,"HI":3,"OR":3,"WA":3
}

# BEA division (8 divisions) as a single integer id 0..7 (includes Mideast explicitly)
BEA_DIVISION_ID = {
    # New England(0)
    "CT":0,"ME":0,"MA":0,"NH":0,"RI":0,"VT":0,
    # Mideast(1)
    "DE":1,"DC":1,"MD":1,"NJ":1,"NY":1,"PA":1,
    # Great Lakes(2)
    "IL":2,"IN":2,"MI":2,"OH":2,"WI":2,
    # Plains(3)
    "IA":3,"KS":3,"MN":3,"MO":3,"NE":3,"ND":3,"SD":3,
    # Southeast(4)
    "AL":4,"AR":4,"FL":4,"GA":4,"KY":4,"LA":4,"MS":4,"NC":4,"SC":4,"TN":4,"VA":4,"WV":4,
    # Southwest(5)
    "AZ":5,"NM":5,"OK":5,"TX":5,
    # Rocky Mountain(6)
    "CO":6,"ID":6,"MT":6,"UT":6,"WY":6,
    # Far West(7)
    "AK":7,"CA":7,"HI":7,"NV":7,"OR":7,"WA":7
}

def _region_ids_from_state(df, state_col="STATE"):
    """Return two numeric columns: region_super_id (0..3) and division_id (0..7)."""
    def _usps(s):
        t = STATE_CODE_MAP.get(int(s), (None,None))
        return t[0]
    usps = df[state_col].map(_usps)
    region_super_id = usps.map(CENSUS_REGION_ID).astype("float64")   # East/Midwest/South/West
    division_id     = usps.map(BEA_DIVISION_ID).astype("float64")    # includes 'Mideast' as id=1
    return pd.DataFrame({"region_super_id": region_super_id, "division_id": division_id}, index=df.index)



# ------------------------------
# 1) Feature construction utils
# ------------------------------

def _build_belt_used(rest_use):
    """Map FARS REST_USE to {0,1,NA}.
    1,2,3,4,5 -> 1 (belt or child restraint; 5=used improperly still indicates using a restraint)
    0 -> 0 (none)
    99 or non-belt codes (helmet etc. = 6,8,13,14,15) -> NA
    """
    # First pass: map the common belt/no-belt codes
    belt = rest_use.map({1:1, 2:1, 3:1, 4:1, 5:1, 0:0})
    # Second pass: overwrite with NaN for codes that are not seat belts or are unknown
    if hasattr(rest_use, "isin"):
        mask_na = rest_use.isin([99, 6, 8, 13, 14, 15])
        belt = belt.mask(mask_na, other=np.nan)
    return belt


def _seat_pos_group(seat_pos):
    # front if 11-19, rear if 21-28; else 'other'
    x = pd.to_numeric(seat_pos, errors="coerce")
    grp = np.where(x//10 == 1, "front", np.where(x//10 == 2, "rear", "other"))
    return pd.Series(grp, index=seat_pos.index if hasattr(seat_pos, "index") else None)

def _hour_sin_cos(hour):
    h = pd.to_numeric(hour, errors="coerce").fillna(-1).astype(int)
    # set invalid hour to 0 for trig (harmless; also masked later by isfinite on C)
    h = h.where((h>=0) & (h<=23), 0)
    ang = 2*np.pi*h/24.0
    return np.sin(ang), np.cos(ang)

def _weekend(day_week):
    # FARS 2000 DAY_WEEK typically 1=Sunday,...,7=Saturday
    d = pd.to_numeric(day_week, errors="coerce")
    return ((d==1) | (d==7)).astype("float64")

def _lgt_flags(lgt):
    # FARS LGT_COND: 1 Daylight; 2 Dark-Not Lighted; 3 Dark-Lighted; 4 Dawn; 5 Dusk; 6 Dark-Unknown
    v = pd.to_numeric(lgt, errors="coerce")
    dark = v.isin([2,3,6]).astype("float64")
    dawn_dusk = v.isin([4,5]).astype("float64")
    return dark, dawn_dusk

def _route_flags(route):
    # FARS ROUTE: 1 Interstate, 2 U.S. Route, 3 State Route, others local/other
    v = pd.to_numeric(route, errors="coerce")
    interstate = (v==1).astype("float64")
    us_or_state = v.isin([2,3]).astype("float64")
    return interstate, us_or_state

def _drinking_flag(drinking):
    # PERSON.DRINKING typically 0=No, 1=Yes, 8/9 unknown
    v = pd.to_numeric(drinking, errors="coerce")
    return (v==1).astype("float64")

# ------------------------------
# 2) Build C, X, Z, Y from raw
# ------------------------------

def load_and_build(in_dir: Path):
    in_dir = Path(in_dir)
    acc = pd.read_csv(in_dir / "ACCIDENT.CSV", low_memory=False)
    per = pd.read_csv(in_dir / "PERSON.CSV", low_memory=False)
    laws = pd.read_csv(in_dir / "state_primary_2000.csv", low_memory=False)

    # --- Person filter: drivers & passengers ---
    per = per[per["PER_TYP"].isin([1,2])].copy()

    # Z: belt_used
    per["belt_used"] = _build_belt_used(per["REST_USE"])

    # Y: died
    per["died"] = (per["INJ_SEV"] == 4).astype(float)

    # Basic covariates
    per["age"]  = pd.to_numeric(per["AGE"], errors="coerce")
    per["male"] = pd.to_numeric(per["SEX"], errors="coerce").map({1:1.0, 2:0.0})
    per.loc[~per["SEX"].isin([1,2]), "male"] = np.nan
    per["is_driver"] = (per["PER_TYP"]==1).astype("float64")
    per["seat_pos_grp"] = _seat_pos_group(per["SEAT_POS"])

    # Accident covariates to merge
    keep_acc_cols = ["ST_CASE","STATE","YEAR","HOUR","DAY_WEEK","LGT_COND","NHS","ROUTE"]
    acc2 = acc[keep_acc_cols].copy()

    # Merge
    df = per.merge(acc2, on="ST_CASE", how="left", validate="m:1")
    
    
    reg_ids = _region_ids_from_state(df, state_col="STATE_x")
    df = pd.concat([df, reg_ids], axis=1)

    # X: primary law (by STATE, year 2000)
    # Expect laws: columns STATE, state_abbr, state_name, primary_2000
    df["X_primary_law"] = df["STATE_x"].astype(int).map(
        laws.set_index("STATE")["primary_2000"].astype(int)
    ).astype(float)

    # Additional covariates for C
    # Seat position dummies
    seat_front = (df["seat_pos_grp"]=="front").astype("float64")
    seat_rear  = (df["seat_pos_grp"]=="rear").astype("float64")
    # Hour sin/cos
    # hour_sin, hour_cos = _hour_sin_cos(df["HOUR_y"])
    hour = pd.to_numeric(df["HOUR_y"], errors="coerce")
    hour = hour.where((hour >= 0) & (hour <= 23), np.nan).astype("float64")
    # Weekend flag
    weekend = _weekend(df["DAY_WEEK"])
    # Light condition flags
    dark, dawn_dusk = _lgt_flags(df["LGT_COND"])
    # Route flags
    interstate, us_or_state = _route_flags(df["ROUTE"])
    # NHS
    on_nhs = pd.to_numeric(df["NHS"], errors="coerce")
    on_nhs = (on_nhs==1).astype("float64")
    # Drinking indicator
    drinking = _drinking_flag(df.get("DRINKING", pd.Series(index=df.index, dtype="float64")))

    # Construct C with ~10-15 dims (no YEAR)
    C = pd.DataFrame({
        "age": df["age"],
        "male": df["male"],
        "is_driver": df["is_driver"],
        "seat_front": seat_front,
        "seat_rear": seat_rear,
        "hour": hour,
        "weekend": weekend,
        "dark": dark,
        "dawn_dusk": dawn_dusk,
        "interstate": interstate,
        "us_or_state": us_or_state,
        "on_nhs": on_nhs,
        "drinking": drinking
        # NEW numeric region features (no one-hot):
        # "region": df["region_super_id"]
        # "division_id": df["division_id"]
    })

    # Final matrices
    X = df["X_primary_law"].astype(float).to_numpy()
    Z = df["belt_used"].astype(float).to_numpy()
    Y = df["died"].astype(float).to_numpy()

    # Force numeric C and arrays
    C = C.apply(pd.to_numeric, errors="coerce").astype("float64")
    C_np = C.to_numpy(dtype="float64")

    # Finite mask
    M = np.isfinite(C_np).all(axis=1) & np.isfinite(X) & np.isfinite(Z) & np.isfinite(Y)
    C_np = C_np[M, :]
    X_np = X[M].astype(int)  # X is 0/1
    Z_np = Z[M].astype(int)  # Z is 0/1
    Y_np = Y[M].astype(float)

    return C_np, X_np, Z_np, Y_np, M, C.columns.tolist()

# ------------------------------
# 3) Diagnostics
# ------------------------------

def quick_diag(X, Z, Y):
    n = len(Y)
    print(f"[diag] n={n:,}")
    # Y binary?
    yvals = np.unique(Y[~np.isnan(Y)])
    print(f"[diag] Y unique values: {yvals[:10]} ... (min={Y.min():.3f}, max={Y.max():.3f})")
    # Z binary?
    zvals = np.unique(Z[~np.isnan(Z)])
    print(f"[diag] Z unique values: {zvals[:10]} ... (min={Z.min():.3f}, max={Z.max():.3f})")
    # X binary?
    xvals = np.unique(X[~np.isnan(X)])
    print(f"[diag] X unique values: {xvals[:10]} ... (min={X.min():.3f}, max={X.max():.3f})")
    # Naive means
    print(f"[diag] mean(Y|X=1)-mean(Y|X=0): { (Y[X==1].mean() - Y[X==0].mean()) }")
    print(f"[diag] mean(Z|X=1)-mean(Z|X=0): { (Z[X==1].mean() - Z[X==0].mean()) }")
    print(f"[diag] mean(Y|Z=1)-mean(Y|Z=0): { (Y[Z==1].mean() - Y[Z==0].mean()) }")

# ------------------------------
# 4) Wire-up to FDCATE.py (same API as your reference script)
# ------------------------------

def run_fd_estimators(C, X, Z, Y, seed=2025, delta=0.0):
    from FDCATE import fit_folds, tau_fd_dr_oof, tau_fd_r_3way_oof_smoothed, tau_naive_oof
    bound_z = [0, 1]
    bound_y = [0, 1]
    folds = fit_folds(C, X, Z, Y, seed)
    tau_dr, tau_dr_model = tau_fd_dr_oof(C, X, Z, Y, folds, delta, bound_y, bound_z, seed)
    tau_r,  tau_r_model  = tau_fd_r_3way_oof_smoothed(C, X, Z, Y, delta, bound_y, bound_z, seed)
    tau_naive = tau_naive_oof(C, X, Z, Y, folds, delta, bound_y, bound_z, seed)
    return tau_dr, tau_r, tau_naive, tau_dr_model, tau_r_model

def summarize(name, tau):
    tau = np.asarray(tau).reshape(-1)
    print(f"[{name}] n={len(tau):,}  mean={np.nanmean(tau):.6f}  std={np.nanstd(tau):.6f}  "
          f"p5={np.nanpercentile(tau,5):.6f}  p50={np.nanpercentile(tau,50):.6f}  p95={np.nanpercentile(tau,95):.6f}")

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
# ------------------------------
# 5) Main
# ------------------------------    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="data/fars", help="Directory containing FARS CSVs and state_primary_2000.csv")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--save_tag", type=str, default="fars_2000", help="Tag for output filenames")
    ap.add_argument("--save_csv", action="store_true", help="Also save the cleaned analytic dataset (optional)")
    args = ap.parse_args()

    in_dir = Path(args.dir)
    assert (in_dir / "ACCIDENT.CSV").exists(), f"Missing {in_dir/'ACCIDENT.CSV'}"
    assert (in_dir / "PERSON.CSV").exists(), f"Missing {in_dir/'PERSON.CSV'}"
    assert (in_dir / "state_primary_2000.csv").exists(), f"Missing {in_dir/'state_primary_2000.csv'}"

    print("[info] Loading and building C, X, Z, Y ...")
    C, X, Z, Y, mask, C_cols = load_and_build(in_dir)

    print("[info] Shapes -> C:", C.shape, " X:", X.shape, " Z:", Z.shape, " Y:", Y.shape)
    quick_diag(X, Z, Y)

    print("[info] Running FD estimators (FD-DR, FD-R, Naive) ...")
    tau_dr, tau_r, tau_naive, tau_dr_model, tau_r_model = run_fd_estimators(C, X, Z, Y, seed=args.seed, delta=args.delta)

    summarize("FD-DR", tau_dr)
    summarize("FD-R ", tau_r)
    summarize("FD-naive", tau_naive)
    print(f"Naive diff E[Y|X=1]-E[Y|X=0] = {Y[X==1].mean() - Y[X==0].mean():.6f}")

    # Save results next to input dir
    out_npz = (in_dir / f"{args.save_tag}.fdcate.npz").as_posix()
    np.savez(out_npz,
             tau_dr=tau_dr, tau_r=tau_r, tau_naive=tau_naive,
             X=X, Z=Z, Y=Y, C_shape=C.shape, C_columns=np.array(C_cols, dtype=object))
    print(f"[info] Saved results to {out_npz}")

    if args.save_csv:
        # Also save an analytic CSV aligning arrays (mask already applied)
        out_csv = in_dir / f"{args.save_tag}.analytic.csv"
        pd.DataFrame(np.column_stack([X, Z, Y, C]),
                     columns=(["X_primary_law","Z_belt_used","Y_died"] + C_cols)).to_csv(out_csv, index=False)
        print(f"[info] Saved cleaned analytic CSV to {out_csv}")


    # --- Further analysis 
    
    import shap
    import matplotlib.pyplot as plt
    
    shap_model_dr = shap.LinearExplainer(tau_dr_model, shap.maskers.Independent(C))
    shap_model_r = shap.LinearExplainer(tau_r_model, shap.maskers.Independent(C))
    
    shap_value_dr = shap_model_dr.shap_values(C)  # C as DataFrame
    shap_value_r = shap_model_r.shap_values(C)  # C as DataFrame
    
    feature_map = {
        1: "age",
        2: "sex",
        3: "is_driver",
        4: "front",
        5: "rear",
        6: "hour",
        7: "weekend",
        8: "dark",
        9: "dawn_dusk", 
        10: "interstate", 
        11: "us_or_state", 
        12: "on_nhs",
        13: "drinking"
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
    VERSION_SAVE = "260924_1800_final"
    plt.savefig(out_dir_fig + "fars_SHAP_analysis" + f"_{VERSION_SAVE}.pdf")
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
    hist_path = out_dir_fig + "fars_histogram" + f"_{VERSION_SAVE}.pdf"

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
    # ax.set_yticks([200_000, 400_000, 600_000, 800_000])
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
    pvc_path = out_dir_fig + "fars_policy_plot" + f"_{VERSION_SAVE}.pdf"

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