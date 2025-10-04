# Debiased Front-Door Learners for Heterogeneous Effects

This repository contains the official implementation of the paper
[**Debiased Front-Door Learners for Heterogeneous Effects**](https://arxiv.org/abs/2509.22531).
 
It includes minimal, runnable code to reproduce (i) the synthetic studies and (ii) the FARS case study reported in the paper.

---

## Repository layout

- `FDCATE.py` — Implementation of **FD‑DR‑Learner** and **FD‑R‑Learner** and a plug‑in baseline (FD‑PI); contains the synthetic experiment driver.
- `analyze_fars_2000_fd.py` — End‑to‑end script that builds the state–year panel for the **FARS primary seat‑belt law** case study and runs FD‑PI / FD‑DR / FD‑R on it.
- `README.md` — This file.
- `data/fars` - A folder containing the the FARS data. 

> **Note.** The estimators, cross‑fitting scheme, XGBoost settings, and stabilization (denominator flooring at 0.05 for inverse weights/density ratios) match the experimental protocol described in the paper.

---

## Quick start

### 1) Environment

We tested with Python 3.9+ on Linux/macOS.

```bash
python -V
# Python 3.9.x or newer
```

Install the minimal dependencies:

```bash
python -m pip install -U pip
python -m pip install numpy pandas scikit-learn xgboost statsmodels matplotlib shap
```

### 2) Reproduce synthetic experiments (Figure 2)

Run:
```bash
python FDCATE.py
```

What it does:
- Generates data under the conditional FD setup.
- Fits **FD‑PI**, **FD‑DR**, and **FD‑R** with cross‑fitting.
- Evaluates RMSE across:
  - sample size sweeps,
  - nuisance “noise” at the $n^{-1/4}$ scale,
  - weak‑overlap stress tests.

Expected outputs:
- Printed metrics (RMSE ± CI) per regime.
- CSVs/plots saved to the working directory (filenames are self‑explanatory and include the regime label).

Tips for determinism:
```bash
PYTHONHASHSEED=0
export PYTHONHASHSEED
# If the scripts expose a --seed flag, set it; otherwise the built‑in defaults are used.
```

### 3) Reproduce the FARS case study (Figure 3)

Run:
```bash
python analyze_fars_2000_fd.py
```

What it does:
- Downloads/loads FARS and NHTSA belt‑use survey tables (public sources) for a **balanced state–year panel**.
- Constructs variables:
  - **Treatment** $X$: primary law in force (state‑year indicator).
  - **Mediator** $Z$: seat‑belt use rate.
  - **Outcome** $Y$: occupant fatality rate (per population or per exposure, depending on availability).
  - **Covariates** $C$: state & year fixed effects and policy‑relevant factors (weather severity, road‑mix, enforcement, driver status, etc.).
- Fits **FD‑PI**, **FD‑DR**, and **FD‑R** and produces:
  - Distributions of $\hat\tau(C)$,
  - Top-$\alpha$ concentration curves,
  - SHAP‑based covariate importance.

Expected outputs:
- Figures and tables in the working directory (histograms of \(\hat\tau\), concentration curves, feature importance).

**Notes on stability**
- Learners use **XGBoost** (50 trees, depth 3, learning rate 0.1, subsample/colsample 0.9) and cross‑fitting consistent with the paper.
- To stabilize finite‑sample variance, **only denominators** that appear in inverse weights/density ratios are floored at **0.05**; numerators are never clipped.

---

## Reproducibility checklist (what we fix to mirror the paper)

- Cross-fitting folds/splits (2-way for FD-PI/FD-DR; 3-way for FD-R’s $b,g,\gamma$ steps).
- XGBoost hyperparameters and linear ridge for the final regression(s).
- $n^{-1/4}$-scale nuisance perturbations used in stress tests.
- Weak-overlap stress by steepening the treatment propensity (no density ratios in FD-R).

<!-- ---

## Citation

If you use this supplement, please cite the paper:

```bibtex
@inproceedings{anonymous2026fdcate,
  title={Heterogeneous Front-Door Effects: Debiased Estimation with Quasi-Oracle Guarantees},
  author={Anonymous},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
``` -->
