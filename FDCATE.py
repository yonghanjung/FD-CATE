import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

import warnings

# Silence only RuntimeWarnings coming from sklearn's LinearModel matmul
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r".*sklearn\.linear_model\._base"
)

# Silence any RuntimeWarning whose message mentions "encountered in matmul"
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*encountered in matmul"
)


# ============================
# Plot colors (vivid as requested)
# ============================
COLOR_NAIVE = "#E41A1C"  # strong red      → Naïve FD
COLOR_FDDR  = "#377EB8"  # strong blue     → FD-DR
COLOR_FDR   = "#4DAF4A"  # strong green    → FD-R
COLOR_FDR3  = "#FFBF00"  # vivid amber    → FD-R3

# ============================
# Utilities & constants
# ============================
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic link used in the DGP."""
    return 1/(1+np.exp(-x))

# --- add these helpers ---
def _zscore_fit(X):
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)   # floor tiny stds
    return mu, sd

def _zscore_apply(X, mu, sd):
    return (X - mu) / sd

def _finite_mask(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a).all(axis=-1) if (a.ndim>1) else np.isfinite(a)
    return m

def _seed_plus(seed, inc=0):
    """
    Coerce `seed` to a Python int and add `inc`.
    Accepts float, numpy scalar, Python int, or None.
    Returns None only if seed is None (leaves RNG to auto-seed).
    """
    if seed is None:
        return None
    try:
        base = int(seed)
    except Exception:
        base = int(np.asarray(seed).item())
    return base + int(inc)

# Theory note (FD-DR): only *denominators* should be positivity-floored.
# (Clipping numerators biases the orthogonal moments; cf. Lemma 2 FDPO.)
PS_FLOOR_DENOM = 0.05 # 1e-8
EPS_RAW = 1e-6  # tiny numeric clip to avoid exact 0/1 in raw weights
EPS_DIV = 1e-8  # floor for denominators in internal residual ratios

def cap01_raw(p: np.ndarray) -> np.ndarray:
    """Clip for probabilities that act as *weights* (close to raw)."""
    return np.clip(p, EPS_RAW, 1-EPS_RAW)

def cap01_denom(p: np.ndarray) -> np.ndarray:
    """Clip for probabilities that appear in *denominators* (FD-DR only)."""
    return np.clip(p, PS_FLOOR_DENOM, 1-PS_FLOOR_DENOM)

def make_three_folds(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three disjoint folds of (approximately) equal size for 3-way cross-fitting."""
    idx = np.arange(n)
    rng.shuffle(idx)
    k = n // 3
    i1 = idx[:k]
    i2 = idx[k:2*k]
    i3 = idx[2*k:]
    return i1, i2, i3

def make_two_folds(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Two disjoint halves (used for cross-fitting; prevents leakage)."""
    idx = np.arange(n); rng.shuffle(idx); mid = n//2
    return idx[:mid], idx[mid:]

def new_xgb_classifier(seed: int) -> XGBClassifier:
    """Binary classifier for eX, q, mZ nuisances."""
    return XGBClassifier(objective="binary:logistic", n_estimators=50, max_depth=3,
                         learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                         reg_lambda=1.0, tree_method="hist", n_jobs=2,
                         random_state=seed, verbosity=0)

def new_xgb_regressor(seed: int) -> XGBRegressor:
    """Regressor for m, mY and FD-R stages (b, g)."""
    return XGBRegressor(objective="reg:squarederror", n_estimators=50, max_depth=3,
                        learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                        reg_lambda=1.0, tree_method="hist", n_jobs=2,
                        random_state=seed, verbosity=0)

# ============================
def _g_features(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Linear feature map for g(X,C). Keeps base code simple and avoids rewriting.
    φ(X,C) = [1, X, zscore(C)].
    """
    mu, sd = _zscore_fit(c)
    cz = _zscore_apply(c, mu, sd)
    ones = np.ones_like(x, dtype=float).reshape(-1,1)
    xcol = x.reshape(-1,1).astype(float)
    return np.hstack([ones, xcol, cz])

# ============================
# DGP
# ============================
@dataclass
class FDParams:
    beta0: float; beta_c: np.ndarray; beta_u: float
    alpha0: float; alpha_c: np.ndarray; alpha_x: float
    theta0: float; theta_c: np.ndarray; theta_z: float; theta_u: float

@dataclass
class SimData:
    C: np.ndarray; X: np.ndarray; Z: np.ndarray; Y: np.ndarray; tau_true: np.ndarray; params: FDParams

def simulate_fd_data_md(n: int = 5000, d: int = 10, seed: int = 123) -> SimData:
    """
    Synthetic front-door DGP with continuous Y and binary X,Z.
    It enforces: E[Y|do(X=1)] > E[Y|do(X=0)] while E[Y|X=1] < E[Y|X=0].
    True τ(C) equals θ_z ⋅ {q(1|1,C) − q(1|0,C)} for binary Z (Eq. (4) specialization).
    """
    rng = np.random.default_rng(seed)
    C = rng.normal(0,1,size=(n,d))
    U = rng.normal(0,1,size=n)

    # Random directions for C's effect, normalized for stable scales.
    wX = rng.normal(0,1,size=d); wX /= np.linalg.norm(wX)+1e-12
    wZ = rng.normal(0,1,size=d); wZ /= np.linalg.norm(wZ)+1e-12
    wY = rng.normal(0,1,size=d); wY /= np.linalg.norm(wY)+1e-12

    # Moderate coefficients → comfortable positivity (no near 0/1 propensities).
    params = FDParams(beta0=0.1, beta_c=0.7*wX, beta_u=0.7,
                      alpha0=0.1, alpha_c=0.7*wZ, alpha_x=1.2,
                      theta0=0.0, theta_c=0.7*wY, theta_z=1.4, theta_u=-2.4)

    # Propensities and draws
    pX = sigmoid(params.beta0 + C@params.beta_c + params.beta_u*U); X = rng.binomial(1,pX)
    pZ = sigmoid(params.alpha0 + C@params.alpha_c + params.alpha_x*X); Z = rng.binomial(1,pZ)

    # Outcome and τ_true(C)
    eps = rng.normal(0,1,size=n)
    Y = params.theta0 + C@params.theta_c + params.theta_z*Z + params.theta_u*U + eps
    b_true   = sigmoid(params.alpha0 + C@params.alpha_c + params.alpha_x) - sigmoid(params.alpha0 + C@params.alpha_c)
    tau_true = params.theta_z * b_true  # τ(C) = θ_z ⋅ b(C)

    # Enforce observational paradox if needed
    if Y[X==1].mean() - Y[X==0].mean() >= 0:
        params.theta_u *= 1.25
        Y = params.theta0 + C@params.theta_c + params.theta_z*Z + params.theta_u*U + eps

    return SimData(C=C,X=X.astype(int),Z=Z.astype(int),Y=Y,tau_true=tau_true,params=params)

# ============================
# Nuisance learners (+ baseline for structural shrinkage)
# ============================
@dataclass
class NuisanceFold:
    eX: XGBClassifier; q: XGBClassifier; m: XGBRegressor; mY: XGBRegressor; mZ: XGBClassifier; mC_lr: LinearRegression

@dataclass
class FoldData:
    train_idx: np.ndarray; test_idx: np.ndarray; nuis: NuisanceFold

def simulate_fd_data_weak_overlap(n: int = 10000, d: int = 10, seed: int = 777,
                                  kappa_e: float = 4.0, kappa_q: float = 1.0) -> SimData:
    """
    Synthetic FD DGP under **weak treatment overlap**: the treatment propensity
    e_X(C,U) is made highly polar through a slope multiplier `kappa_e` on the logit.
    The mediator model remains moderate (no extra multiplier) so that Var(Z|X,C)>0.
    This stresses FD-DR (which uses 1/e weights and density ratios) while leaving
    FD-R comparatively stable (no density ratios).
    """
    rng = np.random.default_rng(seed)
    C = rng.normal(0, 1, size=(n, d))
    U = rng.normal(0, 1, size=n)

    # Random directions (normalized) for stability across dimensions
    wX = rng.normal(0, 1, size=d); wX /= (np.linalg.norm(wX) + 1e-12)
    wZ = rng.normal(0, 1, size=d); wZ /= (np.linalg.norm(wZ) + 1e-12)
    wY = rng.normal(0, 1, size=d); wY /= (np.linalg.norm(wY) + 1e-12)

    # Coefficients
    beta0  = 0.2;  beta_c = 1.2 * wX; beta_u = 1.0  # Treatment logit (scaled by kappa_e below)
    alpha0 = 0.1;  alpha_c = 0.8 * wZ; alpha_x = 1.2  # Mediator logit (kept moderate)
    theta0 = 0.0;  theta_c = 1.0 * wY; theta_z = 1.0; theta_u = 1.0

    # Treatment
    lin_e = beta0 + C @ beta_c + beta_u * U
    e1    = sigmoid(kappa_e * lin_e)  # <-- slope multiplier creates weak overlap
    X     = rng.binomial(1, e1, size=n).astype(int)

    # Mediator
    def q1_given_x(x_arr):
        lin_q = alpha0 + (C @ alpha_c) + alpha_x * x_arr
        return sigmoid(-kappa_q * lin_q)
    q1 = q1_given_x(X)
    Z  = rng.binomial(1, q1, size=n).astype(int)

    # Outcome
    eps = rng.normal(0, 1, size=n)
    Y   = theta0 + C @ theta_c + theta_z * Z + theta_u * U + eps

    # True τ(C) for binary Z (Eq. C.1): θ_z * ( q(1|1,C) - q(1|0,C) )
    q11 = q1_given_x(np.ones(n))
    q10 = q1_given_x(np.zeros(n))
    tau_true = theta_z * (q11 - q10)

    params = FDParams(beta0=beta0, beta_c=beta_c, beta_u=beta_u,
                      alpha0=alpha0, alpha_c=alpha_c, alpha_x=alpha_x,
                      theta0=theta0, theta_c=theta_c, theta_z=theta_z, theta_u=theta_u)
    return SimData(C=C, X=X, Z=Z, Y=Y, tau_true=tau_true, params=params)

def fit_nuisances_fold(C, X, Z, Y, train_idx, seed) -> NuisanceFold:
    """
    Fit nuisances on training fold only (cross-fitting).
    eX(C)=P(X=1|C), q(1|X,C)=P(Z=1|X,C), m(z,x,C)=E[Y|Z=z,X=x,C],
    mY(X,C)=E[Y|X,C], mZ(C)=P(Z=1|C) for FD-R’s b-stage residuals,
    mC_lr(C)=linear E[Y|C] baseline for *structural shrinkage* of m(z,x,C).
    """
    Ctr, Xtr, Ztr, Ytr = C[train_idx], X[train_idx], Z[train_idx], Y[train_idx]
    eX = new_xgb_classifier(seed)
    q  = new_xgb_classifier(seed+1)
    m  = new_xgb_regressor(seed+2)
    mY = new_xgb_regressor(seed+3)
    mZ = new_xgb_classifier(seed+4)

    eX.fit(Ctr, Xtr)
    q.fit(np.column_stack([Xtr, Ctr]), Ztr)
    m.fit(np.column_stack([Ztr, Xtr, Ctr]), Ytr)
    mY.fit(np.column_stack([Xtr, Ctr]), Ytr)
    mZ.fit(Ctr, Ztr)

    # Baseline E[Y|C] to define structural shrinkage for m(z,x,C)
    muC, sdC = _zscore_fit(Ctr)
    Ctr_z = _zscore_apply(Ctr, muC, sdC)
    mC_lr = Ridge(alpha=1e-6).fit(Ctr_z, Ytr)
    # stash the scaler on the model so we can reuse it at predict time
    mC_lr._muC = muC; mC_lr._sdC = sdC
    return NuisanceFold(eX,q,m,mY,mZ,mC_lr)

# --- Structural nuisance degradation (shrink toward baselines) ---
def shrink_prob(p, delta, n_total, rng):
    """
    Additive Gaussian perturbation for probabilities.
    p_new = clip01(p + δ * ε),   ε ~ Normal(mean=n^{-1/4}, var=n^{-1/4})
    Notes:
    - We *do not* center ε at 0; its mean is n^{-1/4} as requested.
    - cap01_raw keeps probabilities in (0,1) to remain usable as weights.
    """
    mean = float(n_total ** (-0.25))
    var  = float(n_total ** (-0.25))   # interpret N(mean, variance)
    std  = float(np.sqrt(var))
    noise = rng.normal(loc=mean, scale=std, size=p.shape)
    return cap01_raw(p + delta * noise)

def shrink_regression(mu, mu_base, delta, n_total, rng):
    """
    Additive Gaussian perturbation for regressions (ignore mu_base).
    mu_new = mu + δ * ε,   ε ~ Normal(mean=n^{-1/4}, var=n^{-1/4})
    """
    mean = float(n_total ** (-0.25))
    var  = float(n_total ** (-0.25))
    std  = float(np.sqrt(var))
    noise = rng.normal(loc=mean, scale=std, size=mu.shape)
    return mu + delta * noise

def nuisance_cache_on(nuis: NuisanceFold, C_eval, X_eval, delta, n_total, rng):
    """
    Compute all nuisances for the fold's *test* points once (coherent cache),
    then apply structural shrinkage δ. Only denominators get a positivity floor.
    """
    # Probabilities used as *weights* (keep raw-like)
    e1    = cap01_raw(nuis.eX.predict_proba(C_eval)[:,1])
    q1_xc = cap01_raw(nuis.q.predict_proba(np.column_stack([X_eval, C_eval]))[:,1])
    
    ones  = np.ones_like(X_eval); zeros = np.zeros_like(X_eval)
    q1_1c = cap01_raw(nuis.q.predict_proba(np.column_stack([ones,  C_eval]))[:,1])
    q1_0c = cap01_raw(nuis.q.predict_proba(np.column_stack([zeros, C_eval]))[:,1])

    m11 = nuis.m.predict(np.column_stack([ones,  ones,  C_eval]))
    m01 = nuis.m.predict(np.column_stack([zeros, ones,  C_eval]))
    m10 = nuis.m.predict(np.column_stack([ones,  zeros, C_eval]))
    m00 = nuis.m.predict(np.column_stack([zeros, zeros, C_eval]))
    
    # 2) Denominators must be independent of δ (freeze here)
    e1_den    = cap01_denom(e1)
    q1_xc_den = cap01_denom(q1_xc)
    
    # 3) Baseline for structural shrinkage of m(z,x,C)
    C_eval_z = _zscore_apply(C_eval, nuis.mC_lr._muC, nuis.mC_lr._sdC)
    mC = nuis.mC_lr.predict(C_eval_z)

    ones  = np.ones_like(X_eval); zeros = np.zeros_like(X_eval)
    q1_1c = cap01_raw(nuis.q.predict_proba(np.column_stack([ones,  C_eval]))[:,1])
    q1_0c = cap01_raw(nuis.q.predict_proba(np.column_stack([zeros, C_eval]))[:,1])
    
    # 4) Apply δ-shrinkage ONLY to numerators / plug-in parts
    e1    = shrink_prob(e1,   delta, n_total, rng)
    q1_xc = shrink_prob(q1_xc,delta, n_total, rng)
    q1_1c = shrink_prob(q1_1c,delta, n_total, rng)
    q1_0c = shrink_prob(q1_0c,delta, n_total, rng)
    m11   = shrink_regression(m11, mC, delta, n_total, rng)
    m01   = shrink_regression(m01, mC, delta, n_total, rng)
    m10   = shrink_regression(m10, mC, delta, n_total, rng)
    m00   = shrink_regression(m00, mC, delta, n_total, rng)

    return dict(
        e1=e1, e0=1-e1, e1_den=e1_den,
        q1_xc=q1_xc, q0_xc=1-q1_xc, q1_xc_den=q1_xc_den,
        q1_1c=q1_1c, q1_0c=q1_0c,
        m11=m11, m01=m01, m10=m10, m00=m00
    )

def m_zx_from_cells(m00,m01,m10,m11, z, x):
    """Compose m(z,x,C) from its four cells for vector z,x ∈ {0,1}^n."""
    return (z*x)*m11 + (z*(1-x))*m10 + ((1-z)*x)*m01 + ((1-z)*(1-x))*m00

# ============================
# Estimators (OOF; cross‑fitted)
# ============================
def tau_naive_oof(C,X,Z,Y,folds,delta,seed):
    """
    Naïve FD plug‑in τ(C) = Σ_{z,x}{q(z|1,C)−q(z|0,C)} e_x(C) m(z,x,C) (Eq. (4)).
    Raw weights only (no denominator clipping here).
    """
    n = len(X); tau = np.empty(n); tau[:] = np.nan
    for k, fold in enumerate(folds):
        rng = np.random.default_rng(seed + 100 + k)
        nuis = fold.nuis; idx = fold.test_idx; c, x = C[idx], X[idx]
        cache = nuisance_cache_on(nuis, c, x, delta, n, rng)
        dq  = cache['q1_1c'] - cache['q1_0c']
        s1  = dq * (cache['m11'] - cache['m01'])
        s0  = dq * (cache['m10'] - cache['m00'])
        tau[idx] = cache['e1'] * s1 + cache['e0'] * s0
    return tau

def tau_fd_dr_oof(C,X,Z,Y,folds,delta,seed):
    """
    FD‑DR via FDPO (Def. 2): φ_1 − φ_0 has E[·|C] = τ(C), doubly robust (Lemma 2).
    We regress ψ := φ1−φ0 on C using OLS. Only *denominators* get clipped.
    """
    n = len(X); tau = np.empty(n); tau[:] = np.nan
    for k, fold in enumerate(folds):
        rng = np.random.default_rng(seed + 200 + k)
        nuis = fold.nuis; idx = fold.test_idx; c, x, z, y = C[idx], X[idx], Z[idx], Y[idx]
        cache = nuisance_cache_on(nuis, c, x, delta, n, rng)

        m_zx = m_zx_from_cells(cache['m00'], cache['m01'], cache['m10'], cache['m11'], z, x)
        r_me_1 = cache['m11']*cache['e1'] + cache['m10']*cache['e0']
        r_me_0 = cache['m01']*cache['e1'] + cache['m00']*cache['e0']
        r_me_z = np.where(z==1, r_me_1, r_me_0)
        nu_meq = r_me_1*cache['q1_xc'] + r_me_0*cache['q0_xc']

        m_z1_x = m_zx_from_cells(cache['m00'], cache['m01'], cache['m10'], cache['m11'], np.ones_like(z), x)
        m_z0_x = m_zx_from_cells(cache['m00'], cache['m01'], cache['m10'], cache['m11'], np.zeros_like(z), x)
        smq_1  = m_z1_x*cache['q1_1c'] + m_z0_x*(1 - cache['q1_1c'])
        smq_0  = m_z1_x*cache['q1_0c'] + m_z0_x*(1 - cache['q1_0c'])

        qz_xc_den = np.where(z==1, cache['q1_xc_den'], 1 - cache['q1_xc_den'])
        xi1 = np.where(z==1, cache['q1_1c'], 1 - cache['q1_1c']) / qz_xc_den
        xi0 = np.where(z==1, cache['q1_0c'], 1 - cache['q1_0c']) / qz_xc_den
        ex_den = np.where(x==1, cache['e1_den'], 1 - cache['e1_den'])
        pi1 = (x==1).astype(float) / ex_den
        pi0 = (x==0).astype(float) / ex_den

        phi1 = xi1*(y - m_zx) + pi1*(r_me_z - nu_meq) + smq_1
        phi0 = xi0*(y - m_zx) + pi0*(r_me_z - nu_meq) + smq_0
        psi  = phi1 - phi0

        model = Ridge(alpha=1e-6)
        mask = np.isfinite(psi).all(axis=-1) if psi.ndim > 1 else np.isfinite(psi)
        if np.any(mask):
            muC, sdC = _zscore_fit(c[mask])
            c_fit = _zscore_apply(c[mask], muC, sdC)
            model.fit(c_fit, psi[mask])
            tau[idx] = model.predict(_zscore_apply(c, muC, sdC))
        else:
            tau[idx] = np.zeros_like(x, dtype=float)
    return tau

def tau_fd_r_oof_smoothed(C,X,Z,Y,folds,delta,seed):
    """
    FD‑R via partial-linear factorization (Prop. 3; Thm. 4): τ(C) = b(C) ⋅ E[g(X,C)|C].
    1) Estimate b(C) by regressing (Z−E[Z|C]) on (X−E[X|C]) with weights (R-learner style).
    2) Estimate g(X,C) by regressing (Y−E[Y|X,C]) on (Z−E[Z|X,C]) with weights.
    3) Pseudo-g: ζ̂(C) = (1−ê)ĝ(0,C)+êĝ(1,C)+(X−ê){ĝ(1,C)−ĝ(0,C)} (Def. 4; Lemma 3).
    4) Output τ̂_R(C) = b̂(C) · ζ̂(C); to stabilize variance, regress b̂·ζ̂ on C and predict.
    """
    n = len(X); tau = np.empty(n); tau[:] = np.nan
    eps = 1e-3
    for k, fold in enumerate(folds):
        rng = np.random.default_rng(seed + 300 + k)
        nuis = fold.nuis; idx = fold.test_idx; c, x, z, y = C[idx], X[idx], Z[idx], Y[idx]
        cache = nuisance_cache_on(nuis, c, x, delta, n, rng)

        # Stage b: effect of X on Z
        mZ = cap01_raw(nuis.mZ.predict_proba(c)[:,1])
        rZ = (z - mZ).astype(float)
        rX = (x - cache['e1']).astype(float)
        mask_b = np.abs(rX) >= eps
        b_model = new_xgb_regressor(seed + 1)
        b_model.fit(c[mask_b], (rZ[mask_b]/rX[mask_b]), sample_weight=(rX[mask_b]**2))
        b_hat = b_model.predict(c)

        # Stage g: effect of Z on Y given (X,C)
        eZ = cache['q1_xc']
        mY = nuis.mY.predict(np.column_stack([x, c]))
        rY = (y - mY).astype(float)
        rZ2= (z - eZ).astype(float)
        safe = np.abs(rZ2) >= EPS_DIV
        mask_g = np.abs(rZ2) >= eps
        g_model = new_xgb_regressor(seed + 2)
        if np.any(safe):
            Xg = np.column_stack([x[safe], c[safe]])
            yg = rY[safe] / rZ2[safe]
            wg = (rZ2[safe]**2)
            g_model.fit(Xg, yg, sample_weight=wg)
        else:
            # Degenerate case: no stable points; fit a trivial constant model
            Xg = np.column_stack([x, c])
            yg = np.zeros_like(x, dtype=float)
            g_model.fit(Xg, yg)
        # g_model.fit(np.column_stack([x[mask_g], c[mask_g]]), (rY[mask_g]/rZ2[mask_g]), sample_weight=(rZ2[mask_g]**2))
        g1 = g_model.predict(np.column_stack([np.ones_like(x), c]))
        g0 = g_model.predict(np.column_stack([np.zeros_like(x), c]))

        # Pseudo-g and composition; final smoothing ≈ E[b̂·ζ̂|C] #
        zeta = (1 - cache['e1'])*g0 + cache['e1']*g1 + (x - cache['e1'])*(g1 - g0)
        target_tau = b_hat * zeta
        lin = Ridge(alpha=1e-6)
        mask = np.isfinite(target_tau)
        if np.any(mask):
            muC, sdC = _zscore_fit(c[mask])
            lin.fit(_zscore_apply(c[mask], muC, sdC), target_tau[mask])
            tau[idx] = lin.predict(_zscore_apply(c, muC, sdC))
        else:
            tau[idx] = np.zeros_like(x, dtype=float)
    return tau

def _ridge_solve(Xmat: np.ndarray, y: np.ndarray, alpha: float=1e-6) -> np.ndarray:
    """Closed-form ridge: (X^T X + αI)^{-1} X^T y."""
    d = Xmat.shape[1]
    XtX = Xmat.T @ Xmat
    XtX.flat[::d+1] += alpha
    Xty = Xmat.T @ y
    try:
        theta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(XtX) @ Xty
    return theta

def tau_fd_r_3way_oof_smoothed(C,X,Z,Y,delta: float,seed: int, g_solver: str="direct", swap_average: bool=True) -> np.ndarray:
    """
    FD-R with 3-way cross-fitting and ζ-regression (Eq. 35).
    - Splits data into (D1,D2,D3).
    - Fit nuisances on D1, fit (b,g) on D2 (using either 'direct' or 'ratio' solver for g),
      construct ζ on D3 and regress ζ ~ C to estimate γ(C). Return τ̂(C) = b̂(C)·γ̂(C).
    - If swap_average=True, swap D1 and D2 and average the two predictions on each D3 fold.
    """
    n = len(X); tau_hat = np.empty(n); tau_hat[:] = np.nan
    rng = np.random.default_rng(_seed_plus(seed, 700))
    i1, i2, i3 = make_three_folds(n, rng)
    # Define the two permutations per test fold k (nuis,bg,test)
    perms = {
        1: [(i1,i2,i3),(i2,i1,i3)],
        2: [(i2,i3,i1),(i3,i2,i1)],
        3: [(i3,i1,i2),(i1,i3,i2)],
    }
    # Map fold indices to integer label
    fold_label = np.empty(n, dtype=int); fold_label[i1]=1; fold_label[i2]=2; fold_label[i3]=3

    # For each test fold k, compute prediction twice with (D1,D2) swapped and average
    for k_label, two_perms in perms.items():
        preds = []
        for (D1,D2,D3) in (two_perms if swap_average else two_perms[:1]):
            # 1) Fit nuisances on D1
            nuis = fit_nuisances_fold(C,X,Z,Y,train_idx=D1,seed=_seed_plus(seed, 10 + k_label))
            # 2) Learn b and g on D2
            c2, x2, z2, y2 = C[D2], X[D2], Z[D2], Y[D2]
            cache2 = nuisance_cache_on(nuis, c2, x2, delta, n, rng)
            # b-stage (R-learner style): regress rZ on rX
            rZ = (z2 - nuis.mZ.predict(c2)).astype(float)
            rX = (x2 - cache2['e1']).astype(float)
            mask_b = np.abs(rX) >= 1e-3
            # Use XGB for b to keep base behavior
            b_model = new_xgb_regressor(_seed_plus(seed, 111 + k_label))
            if np.any(mask_b):
                b_model.fit(c2[mask_b], (rZ[mask_b]/rX[mask_b]), sample_weight=(rX[mask_b]**2))
            else:
                b_model.fit(c2, np.zeros_like(x2, dtype=float))
            # g-stage
            eZ = cache2['q1_xc']
            mY = nuis.mY.predict(np.column_stack([x2, c2]))
            rY = (y2 - mY).astype(float)
            rZ2= (z2 - eZ).astype(float)
            if g_solver == "direct":
                # Direct optimization: min Σ (rY − rZ2 * g(X,C))^2 via ridge on φ' = rZ2*φ
                Phi = _g_features(x2, c2)
                Xmat = (rZ2.reshape(-1,1) * Phi)
                theta = _ridge_solve(Xmat, rY, alpha=1e-4)
                # Predict g on D2 points for later ζ on D3 (we need g(1,C) and g(0,C) on D3)
                # We'll store theta and the z-score params are baked into _g_features each time.
                def predict_g(x_in, c_in):
                    Phi_in = _g_features(x_in, c_in)
                    return Phi_in @ theta
                g_predict = predict_g
            else:
                # Ratio/weights formulation (original): minimize Σ rZ2^2 (rY/rZ2 − g)^2
                g_model = new_xgb_regressor(_seed_plus(seed, 222 + k_label))
                safe = np.abs(rZ2) >= EPS_DIV
                if np.any(safe):
                    Xg = np.column_stack([x2[safe], c2[safe]])
                    yg = rY[safe] / rZ2[safe]
                    wg = (rZ2[safe]**2)
                    g_model.fit(Xg, yg, sample_weight=wg)
                else:
                    Xg = np.column_stack([x2, c2]); yg = np.zeros_like(x2, dtype=float)
                    g_model.fit(Xg, yg)
                def g_predict(x_in, c_in):
                    return g_model.predict(np.column_stack([x_in, c_in]))
            # 3) On D3: build ζ, regress ζ ~ C to get γ̂, and output τ̂ = b̂·γ̂
            c3, x3, z3, y3 = C[D3], X[D3], Z[D3], Y[D3]
            cache3 = nuisance_cache_on(nuis, c3, x3, delta, n, rng)
            b_hat = b_model.predict(c3)
            g1 = g_predict(np.ones_like(x3), c3)
            g0 = g_predict(np.zeros_like(x3), c3)
            zeta = (1 - cache3['e1'])*g0 + cache3['e1']*g1 + (x3 - cache3['e1'])*(g1 - g0)
            # γ̂(C) = E[ζ | C] via ridge on z-scored C (Eq. 35)
            target_tau = b_hat * zeta
            lin = Ridge(alpha=1e-6)
            mask = np.isfinite(target_tau)
            if np.any(mask):
                muC, sdC = _zscore_fit(c3[mask])
                lin.fit(_zscore_apply(c3[mask], muC, sdC), target_tau[mask])
                preds.append( lin.predict(_zscore_apply(c3, muC, sdC)) )
            else:
                preds.append( np.zeros_like(x3, dtype=float) )
        tau_hat[D3] = np.mean(np.vstack(preds), axis=0)
    return tau_hat

# ============================
# Runner + statistics
# ============================
def fit_folds(C,X,Z,Y,seed: int) -> List[FoldData]:
    """Make two cross‑fitting folds and fit nuisances on each training half."""
    rng = np.random.default_rng(seed+99); i1, i2 = make_two_folds(len(X), rng)
    f1 = FoldData(train_idx=i1, test_idx=i2, nuis=fit_nuisances_fold(C,X,Z,Y,i1,seed+1))
    f2 = FoldData(train_idx=i2, test_idx=i1, nuis=fit_nuisances_fold(C,X,Z,Y,i2,seed+1000))
    return [f1, f2]

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square error between τ̂(C) and τ(C)."""
    return float(np.sqrt(np.mean((a-b)**2)))

def run_one_n(n: int, d: int, R: int = 4, noise_coeff: float = 0.0, mode: str = "rate",
              base_seed: int = 2025, verbose: bool = False) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    R replications at fixed (n,d). Structural shrinkage δ:
      • if mode=="rate": δ = noise_coeff * n^{-1/4}  (rate-style noise),
      • else (mode=="abs"): δ = noise_coeff         (absolute misspecification).
    """
    rms_naive=[]; rms_dr=[]; rms_r=[]; rms_r3=[]
    for r in range(R):
        seed = base_seed + 67*r + n
        data = simulate_fd_data_md(n=n,d=d,seed=seed)
        C,X,Z,Y,tau_true = data.C, data.X, data.Z, data.Y, data.tau_true
        if verbose and r==0:
            do = tau_true.mean(); obs = Y[X==1].mean()-Y[X==0].mean()
            print(f"n={n}, mode={mode}, noise_coeff={noise_coeff}: do-diff={do:.3f}, obs-diff={obs:.3f}")
        folds = fit_folds(C,X,Z,Y,seed)
        delta = noise_coeff
        tau_naive = tau_naive_oof(C,X,Z,Y,folds,delta,seed+10)
        tau_dr    = tau_fd_dr_oof(C,X,Z,Y,folds,delta,seed+20)
        tau_r     = tau_fd_r_oof_smoothed(C,X,Z,Y,folds,delta,seed+30)
        tau_r3     = tau_fd_r_3way_oof_smoothed(C,X,Z,Y,delta,seed+40,g_solver="direct",swap_average=True)
        rms_naive.append(rmse(tau_naive, tau_true))
        rms_dr.append(rmse(tau_dr, tau_true))
        rms_r.append(rmse(tau_r, tau_true))
        rms_r3.append(rmse(tau_r3, tau_true))
    return np.array(rms_naive), np.array(rms_dr), np.array(rms_r), np.array(rms_r3)

def mean_ci(vals: np.ndarray) -> Tuple[float,float]:
    """Mean and 95% CI half‑width (normal approximation) across Monte‑Carlo replications."""
    R = len(vals); m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if R>1 else 0.0
    hw = 1.96 * s / np.sqrt(R) if R>1 else 0.0
    return m, hw

# ============================
# Three simulations FIRST (no plotting here)
# ============================
def run_three_simulations(ns_list, d, R, noise_abs_for_n, noise_grid_for_fixed_n, fixed_n, mode):
    """
    Sim‑1: δ=0 across n-grid (no structural error).
    Sim‑2: δ=noise_abs_for_n across n-grid (moderate misspecification).
    Sim‑3: fixed n; sweep δ over noise_grid_for_fixed_n.
    Returns DataFrames with mean±CI per method.
    """
    # --- Sim‑1
    rows = []
    for n in ns_list:
        rn, rd, rr, rr3 = run_one_n(n, d, R=R, noise_coeff=0.0, mode=mode, base_seed=6060, verbose=(n==ns_list[0]))
        mN,hN = mean_ci(rn); mD,hD = mean_ci(rd); mR,hR = mean_ci(rr); mR3,hR3 = mean_ci(rr3)
        rows.append((n,mN,hN,mD,hD,mR,hR,mR3,hR3))
    tab_n0 = pd.DataFrame(rows, columns=["n","Naive_mean","Naive_hw","FDDR_mean","FDDR_hw","FDR_mean","FDR_hw", "FDR3_mean", "FDR3_hw"])

    # --- Sim‑2
    rows = []
    for n in ns_list:
        rn, rd, rr, rr3 = run_one_n(n, d, R=R, noise_coeff=noise_abs_for_n, mode=mode, base_seed=7070, verbose=(n==ns_list[0]))
        mN,hN = mean_ci(rn); mD,hD = mean_ci(rd); mR,hR = mean_ci(rr); mR3,hR3 = mean_ci(rr3)
        rows.append((n,mN,hN,mD,hD,mR,hR,mR3,hR3))
    tab_nh = pd.DataFrame(rows, columns=["n","Naive_mean","Naive_hw","FDDR_mean","FDDR_hw","FDR_mean","FDR_hw", "FDR3_mean", "FDR3_hw"])

    # --- Sim‑3
    rows = []
    for coeff in noise_grid_for_fixed_n:
        rn, rd, rr, rr3 = run_one_n(fixed_n, d, R=R, noise_coeff=coeff, mode=mode, base_seed=8080, verbose=(coeff==noise_grid_for_fixed_n[0]))
        mN,hN = mean_ci(rn); mD,hD = mean_ci(rd); mR,hR = mean_ci(rr); mR3,hR3 = mean_ci(rr3)
        rows.append((coeff,mN,hN,mD,hD,mR,hR,mR3,hR3))
    tab_noise = pd.DataFrame(rows, columns=["delta","Naive_mean","Naive_hw","FDDR_mean","FDDR_hw","FDR_mean","FDR_hw", "FDR3_mean","FDR3_hw"])
    return tab_n0, tab_nh, tab_noise

# ============================
# Plotting LATER (after sims have finished)
# ============================
def plot_rmse_vs_n_with_ci(tab: pd.DataFrame, title: str):
    """Plot RMSE vs n with mean ± 95% CI error bars for each method."""
    n = tab["n"].values
    plt.figure()
    plt.errorbar(n, tab["Naive_mean"], yerr=tab["Naive_hw"], marker='o', linewidth=2.5, capsize=4, label="Naive FD", color=COLOR_NAIVE)
    plt.errorbar(n, tab["FDDR_mean"],  yerr=tab["FDDR_hw"],  marker='s', linewidth=2.5, capsize=4, label="FD-DR",   color=COLOR_FDDR)
    plt.errorbar(n, tab["FDR_mean"],   yerr=tab["FDR_hw"],   marker='^', linewidth=2.5, capsize=4, label="FD-R",    color=COLOR_FDR)
    plt.errorbar(n, tab["FDR3_mean"],   yerr=tab["FDR3_hw"], marker='*', linewidth=2.5, capsize=4, label="FD-R3",    color=COLOR_FDR3)
    plt.xlabel("Sample size n"); plt.ylabel("RMSE (mean ± 95% CI)")
    plt.title(title); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

def plot_rmse_vs_delta_with_ci(tab: pd.DataFrame, n_for_title: int):
    """Plot RMSE vs structural noise δ (fixed n) with mean ± 95% CI error bars."""
    dlt = tab["delta"].values
    plt.figure()
    plt.errorbar(dlt, tab["Naive_mean"], yerr=tab["Naive_hw"], marker='o', linewidth=2.5, capsize=4, label="Naive FD", color=COLOR_NAIVE)
    plt.errorbar(dlt, tab["FDDR_mean"],  yerr=tab["FDDR_hw"],  marker='s', linewidth=2.5, capsize=4, label="FD-DR",   color=COLOR_FDDR)
    plt.errorbar(dlt, tab["FDR_mean"],   yerr=tab["FDR_hw"],   marker='^', linewidth=2.5, capsize=4, label="FD-R",    color=COLOR_FDR)
    plt.errorbar(dlt, tab["FDR3_mean"],   yerr=tab["FDR3_hw"],   marker='*', linewidth=2.5, capsize=4, label="FD-R3",    color=COLOR_FDR3)
    plt.xlabel("Structural nuisance shrinkage δ"); plt.ylabel("RMSE (mean ± 95% CI)")
    plt.title(f"FD CATE RMSE vs δ (n={n_for_title})"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ============================
# Execute: run 3 simulations first, then draw plots
# ============================

def run_weak_overlap_simulation(n: int, d: int, R: int, kappa_e_grid: List[float],
                                base_seed: int = 8080) -> pd.DataFrame:
    """
    Sim‑4: Weak overlap — fixed n, vary severity via slope multiplier kappa_e.
    For each kappa_e, run R replications and report RMSE mean ± 95% CI.
    """
    rows = []
    for kappa_e in kappa_e_grid:
        rms_naive=[]; rms_dr=[]; rms_r=[]; rms_r3=[]
        for r in range(R):
            seed = base_seed + 97*r + int(10*kappa_e) + n
            data = simulate_fd_data_weak_overlap(n=n, d=d, seed=seed, kappa_e=kappa_e, kappa_q=kappa_e)
            C,X,Z,Y,tau_true = data.C, data.X, data.Z, data.Y, data.tau_true
            folds = fit_folds(C,X,Z,Y,seed)
            delta = 0.0  # no structural shrinkage in this sim; challenge is weak overlap only
            tau_naive = tau_naive_oof(C,X,Z,Y,folds,delta,seed+10)
            tau_dr    = tau_fd_dr_oof(C,X,Z,Y,folds,delta,seed+20)
            tau_r     = tau_fd_r_oof_smoothed(C,X,Z,Y,folds,delta,seed+30)
            tau_r3    = tau_fd_r_3way_oof_smoothed(C,X,Z,Y,delta,seed+30,g_solver="direct",swap_average=True)
            rms_naive.append(rmse(tau_naive, tau_true))
            rms_dr.append(rmse(tau_dr, tau_true))
            rms_r.append(rmse(tau_r, tau_true))
            rms_r3.append(rmse(tau_r3, tau_true))
        mN,hN = mean_ci(np.array(rms_naive)); mD,hD = mean_ci(np.array(rms_dr)); mR,hR = mean_ci(np.array(rms_r)); mR3,hR3 = mean_ci(np.array(rms_r3))
        rows.append((kappa_e,mN,hN,mD,hD,mR,hR,mR3,hR3))
    tab = pd.DataFrame(rows, columns=["kappa_e","Naive_mean","Naive_hw","FDDR_mean","FDDR_hw","FDR_mean","FDR_hw", "FDR3_mean", "FDR3_hw"])
    return tab

def plot_rmse_vs_overlap_with_ci(tab: pd.DataFrame, n_for_title: int):
    """Plot RMSE vs weak-overlap severity κ_e (fixed n) with mean ± 95% CI error bars."""
    x = tab["kappa_e"].values
    plt.figure()
    plt.errorbar(x, tab["Naive_mean"], yerr=tab["Naive_hw"], marker='o', linewidth=2.5, capsize=4, label="Naive FD", color=COLOR_NAIVE)
    plt.errorbar(x, tab["FDDR_mean"],  yerr=tab["FDDR_hw"],  marker='s', linewidth=2.5, capsize=4, label="FD-DR",   color=COLOR_FDDR)
    plt.errorbar(x, tab["FDR_mean"],   yerr=tab["FDR_hw"],   marker='^', linewidth=2.5, capsize=4, label="FD-R",    color=COLOR_FDR)
    plt.errorbar(x, tab["FDR3_mean"],   yerr=tab["FDR3_hw"],   marker='*', linewidth=2.5, capsize=4, label="FD-R3",    color=COLOR_FDR3)
    plt.xlabel("Weak-overlap severity κ_e  (larger = more extreme propensities)")
    plt.ylabel("RMSE (mean ± 95% CI)")
    plt.title(f"FD CATE RMSE vs Weak Overlap (n={n_for_title})")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ============================
# Execute: run 3 simulations first, then draw plots
# ============================
if __name__ == "__main__":
    # (Smaller grid for responsiveness here; increase for paper‑grade results.)
    NS = [1000, 2500, 5000, 10000, 20000, 50000]
    DIM = 10
    ROUNDS = 5
    DELTA_ABS_FOR_N = 0.5
    DELTA_GRID_FIXED_N = [0.0, 0.15, 0.3, 0.45, 0.6, 0.85, 1.0]
    FIXED_N_FOR_SWEEP = 5000
    MODE = "rate" # "abs" or "rate"

    # --- Run simulations (no plotting yet) ---
    tab_n0, tab_nh, tab_noise = run_three_simulations(
        NS, DIM, ROUNDS, DELTA_ABS_FOR_N, DELTA_GRID_FIXED_N, FIXED_N_FOR_SWEEP, MODE
    )
    
    # --- Sim‑4: Weak Overlap (new) ---
    SEVERITY_GRID = [2.0, 4.0, 6.0, 8.0, 10.0]  # κ_e multipliers
    FIXED_N_WEAK = FIXED_N_FOR_SWEEP  # reuse same n as Sim‑3 unless changed above
    tab_weak = run_weak_overlap_simulation(FIXED_N_WEAK, DIM, ROUNDS, SEVERITY_GRID, base_seed=9090)
    print(tab_weak)

    # --- Draw plots only after all sims complete ---
    plot_rmse_vs_n_with_ci(tab_n0,  "FD CATE RMSE vs n — δ=0 (no structural nuisance error)")
    plot_rmse_vs_n_with_ci(tab_nh,  f"FD CATE RMSE vs n — δ={DELTA_ABS_FOR_N} (structural nuisance shrinkage)")
    plot_rmse_vs_delta_with_ci(tab_noise, FIXED_N_FOR_SWEEP)
    plot_rmse_vs_overlap_with_ci(tab_weak, FIXED_N_WEAK)

    # Optional: print the numeric summaries (means ± CI half‑widths)
    print(tab_n0); print(tab_nh); print(tab_noise)
