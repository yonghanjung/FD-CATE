# plot_fdcate_results.py
# Usage:
#   python plot_fdcate_results.py \
#       --npz data/tnp/iu3g-qa69.tipped_binary.fdcate.npz \
#       --outdir figs_tnp_2019

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _finite_mask(*arrs):
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def _decile_bins(score, n_bins=10):
    qs = np.quantile(score, np.linspace(0, 1, n_bins + 1))
    edges = [qs[0]]
    for v in qs[1:]:
        if v <= edges[-1]:
            v = edges[-1] + 1e-12
        edges.append(v)
    return np.array(edges)


def plot_hist(tau, name, outdir):
    m = np.isfinite(tau)
    v = tau[m]
    if v.size == 0:
        return
    mean_v = v.mean()
    fig = plt.figure()
    plt.hist(v, bins=60)
    plt.axvline(mean_v, linestyle='--')
    plt.title(f"Distribution of $\hat\\tau$ ({name})\nmean={mean_v:.6f}, n={v.size:,}")
    plt.xlabel("$\hat\\tau$")
    plt.ylabel("count")
    plt.tight_layout()
    fig.savefig(Path(outdir) / f"hist_{name}.png", dpi=160)
    plt.close(fig)


def plot_ecdf_compare(tau_a, tau_b, name_a, name_b, outdir):
    m = np.isfinite(tau_a) & np.isfinite(tau_b)
    a = np.sort(tau_a[m])
    b = np.sort(tau_b[m])
    if a.size == 0 or b.size == 0:
        return
    fig = plt.figure()
    x_a = np.unique(a)
    y_a = np.searchsorted(a, x_a, side="right") / a.size
    x_b = np.unique(b)
    y_b = np.searchsorted(b, x_b, side="right") / b.size
    plt.plot(x_a, y_a, label=name_a)
    plt.plot(x_b, y_b, label=name_b)
    plt.title("ECDF of $\hat\\tau$")
    plt.xlabel("$\hat\\tau$")
    plt.ylabel("ECDF")
    plt.legend()
    plt.tight_layout()
    fig.savefig(Path(outdir) / "ecdf_compare.png", dpi=160)
    plt.close(fig)


def plot_qq(tau_a, tau_b, name_a, name_b, outdir):
    m = np.isfinite(tau_a) & np.isfinite(tau_b)
    a = np.sort(tau_a[m])
    b = np.sort(tau_b[m])
    n = min(a.size, b.size)
    if n == 0:
        return
    qa = a[np.linspace(0, n - 1, n, dtype=int)]
    qb = b[np.linspace(0, n - 1, n, dtype=int)]
    fig = plt.figure()
    plt.scatter(qa, qb, s=4, alpha=0.6)
    lims = [min(qa.min(), qb.min()), max(qa.max(), qb.max())]
    plt.plot(lims, lims, linestyle='--')
    plt.title(f"Q–Q: {name_a} vs {name_b}")
    plt.xlabel(f"{name_a} quantiles")
    plt.ylabel(f"{name_b} quantiles")
    plt.tight_layout()
    fig.savefig(Path(outdir) / "qq_compare.png", dpi=160)
    plt.close(fig)


def plot_policy_value_curve(tau_dict, outdir):
    # Average τ̂ among top-α fraction (α ∈ [0,1]).
    fig = plt.figure()
    alphas = np.linspace(0.01, 1.0, 100)
    for name, tau in tau_dict.items():
        v = tau[np.isfinite(tau)]
        v = np.sort(v)[::-1]
        if v.size == 0:
            continue
        cumsum = np.cumsum(v)
        k = np.maximum(1, (alphas * len(v)).astype(int))  # ensure >=1
        means = cumsum[k - 1] / k
        plt.plot(alphas, means, label=name)
    plt.title("Policy value curve: mean $\\hat\\tau$ among top-α")
    plt.xlabel("α (fraction targeted by largest $\\hat\\tau$)")
    plt.ylabel("mean $\\hat\\tau$ in top-α")
    plt.legend()
    plt.tight_layout()
    fig.savefig(Path(outdir) / "policy_value_curve.png", dpi=160)
    plt.close(fig)


def plot_calibration_by_decile(tau, X, Y, name, outdir, n_bins=10):
    # Diagnostic only: within τ̂-deciles, plot naive Δ̄Y = E[Y|X=1] − E[Y|X=0]
    m = _finite_mask(tau, X, Y)
    s, x, y = tau[m], X[m], Y[m]
    if s.size == 0:
        return
    edges = _decile_bins(s, n_bins=n_bins)
    centers, deltas, sizes = [], [], []
    for i in range(len(edges) - 1):
        mask = (s >= edges[i]) & (s < edges[i + 1])
        if mask.sum() < 20:
            continue
        centers.append(np.nanmean(s[mask]))
        y1 = y[mask & (x == 1)]
        y0 = y[mask & (x == 0)]
        d = (np.nanmean(y1) - np.nanmean(y0)) if (y1.size > 0 and y0.size > 0) else np.nan
        deltas.append(d)
        sizes.append(mask.sum())
    centers = np.array(centers)
    deltas = np.array(deltas)
    sizes = np.array(sizes)

    fig = plt.figure()
    plt.scatter(centers, deltas, s=10 + 90 * (sizes / sizes.max()), alpha=0.8)
    plt.axhline(0.0, linestyle='--')
    plt.title(f'Naive contrast by $\\hat\\tau$ decile ({name})')
    plt.xlabel('mean $\\hat\\tau$ in decile')
    plt.ylabel('$E[Y|X=1]-E[Y|X=0]$ (naive)')
    plt.tight_layout()
    fig.savefig(Path(outdir) / f"calibration_deciles_{name}.png", dpi=160)
    plt.close(fig)


def plot_fd_diagnostics_by_decile(tau_ref, X, Z, outdir, n_bins=10):
    # Using one τ̂ (e.g., FD-R) to bin; report:
    # 1) treated share P(X=1)
    # 2) P(Z=1|X=1)
    # 3) count of violations #(Z=1 & X=0)
    m = _finite_mask(tau_ref, X, Z)
    s, x, z = tau_ref[m], X[m], Z[m]
    if s.size == 0:
        return
    edges = _decile_bins(s, n_bins=n_bins)

    centers, px1, pz1_x1, viol = [], [], [], []
    for i in range(len(edges) - 1):
        mask = (s >= edges[i]) & (s < edges[i + 1])
        if mask.sum() < 20:
            continue
        centers.append(np.nanmean(s[mask]))
        xm = x[mask]
        zm = z[mask]
        px1.append(np.mean(xm == 1))
        denom = max(1, int(np.sum(xm == 1)))
        pz1_x1.append(np.sum((xm == 1) & (zm == 1)) / denom)
        viol.append(int(np.sum((xm == 0) & (zm == 1))))

    centers = np.array(centers)
    fig = plt.figure()
    plt.plot(centers, px1, marker='o')
    plt.title('Overlap diagnostic by $\\hat\\tau$ decile (treated share)')
    plt.xlabel('mean $\\hat\\tau$ in decile')
    plt.ylabel('P(X=1)')
    plt.tight_layout()
    fig.savefig(Path(outdir) / "fd_diag_treated_share.png", dpi=160)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(centers, pz1_x1, marker='o')
    plt.title('Front-door path strength by $\\hat\\tau$ decile: P(Z=1 | X=1)')
    plt.xlabel('mean $\\hat\\tau$ in decile')
    plt.ylabel('P(Z=1 | X=1)')
    plt.tight_layout()
    fig.savefig(Path(outdir) / "fd_diag_pZ1_given_X1.png", dpi=160)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(centers, viol, marker='o')
    plt.title('One-sided compliance violations by $\\hat\\tau$ decile: #(Z=1 & X=0)')
    plt.xlabel('mean $\\hat\\tau$ in decile')
    plt.ylabel('count')
    plt.tight_layout()
    fig.savefig(Path(outdir) / "fd_diag_violations.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to *.fdcate.npz emitted by analyze_tnp_2019_fd.py")
    ap.add_argument("--outdir", default="figs_fdcate", help="Directory to save figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz)
    tau_dr = data["tau_dr"]
    tau_r  = data["tau_r"]
    tau_nv = data["tau_naive"]
    X, Z, Y = data["X"], data["Z"], data["Y"]

    # Quick text summary
    def _summ(name, v):
        m = np.isfinite(v)
        w = v[m]
        if w.size:
            print(f"{name}: n={w.size:,} mean={w.mean():.6f} q10={np.quantile(w,.1):.6f} "
                  f"median={np.quantile(w,.5):.6f} q90={np.quantile(w,.9):.6f}")
        else:
            print(f"{name}: no finite values")
    _summ("FD-DR", tau_dr)
    _summ("FD-R ", tau_r)
    _summ("FD-naive(τ)", tau_nv)

    # 1) Marginal τ̂ distributions
    plot_hist(tau_dr, "FD-DR", outdir)
    plot_hist(tau_r,  "FD-R",  outdir)
    plot_hist(tau_nv, "FD-naive", outdir)

    # 2) ECDF and Q–Q comparisons (FD-DR vs FD-R)  <-- FIXED: pass outdir
    plot_ecdf_compare(tau_dr, tau_r, "FD-DR", "FD-R", outdir)
    plot_qq(tau_dr, tau_r, "FD-DR", "FD-R", outdir)

    # 3) Policy value curve
    tau_dict = {"FD-DR": tau_dr, "FD-R": tau_r}
    plot_policy_value_curve(tau_dict, outdir)

    # 4) “Calibration” diagnostics by τ̂-deciles (naive contrast)
    plot_calibration_by_decile(tau_dr, X, Y, "FD-DR", outdir)
    plot_calibration_by_decile(tau_r,  X, Y, "FD-R",  outdir)

    # 5) Front-door diagnostics by τ̂-deciles (use FD-R bins)
    plot_fd_diagnostics_by_decile(tau_r, X, Z, outdir)

    print(f"Saved figures in: {outdir.resolve()}")

if __name__ == "__main__":
    main()
