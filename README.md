# FD-CATE

Front-door CATE/ATE estimation toolkit with paper-parity defaults for debiased front-door learners.

This repository keeps the original research scripts (`FDCATE.py`, `analyze_fars_2000_fd.py`) and adds a standard-library interface (`fd_cate`) with a stable artifact contract.

## Install

```bash
python -m pip install -U pip
python -m pip install .
```

Default learner is `xgb` (XGBoost). `nn` is also supported via `nuisance_learner="nn"`.

## Quickstart (Python API)

```python
from fd_cate import FDCATE
from FDCATE import simulate_fd_data_md

# synthetic example
D = simulate_fd_data_md(n=500, d=10, seed=0)

est = FDCATE(method="fd-dr", nuisance_learner="xgb", random_state=0)
est.fit(D.C, D.Y, t=D.X, m=D.Z)

tau = est.effect(D.C)
print(est.ate_)
print(est.summary())
```

## Quickstart (CLI)

```bash
# generate synthetic csv
fdcate synthetic --n 300 --d 8 --seed 42 --out synthetic.csv

# fit + write standard artifacts
fdcate fit \
  --data synthetic.csv \
  --outcome y --treat t --med m \
  --outdir out/

# diagnostics only
fdcate doctor \
  --data synthetic.csv \
  --outcome y --treat t --med m
```

Standard artifacts under `out/`:
- `summary.txt`
- `results.json`
- `diagnostics.json`
- `effects.csv`
- `model.pkl`

## Model Compatibility Policy (`model.pkl`)

`model.pkl` loading is allowed only when **major.minor** package versions match.

- Example: model saved with `0.1.x` can be loaded by `0.1.y`.
- Example: model saved with `0.1.x` cannot be loaded by `0.2.x`.

## Scope (v0.1)

Supported:
- binary treatment `T ∈ {0,1}`
- binary mediator `M ∈ {0,1}`
- numeric covariates
- continuous or binary outcome (regression handling)

Not supported:
- non-binary `T`/`M`
- automatic categorical encoding pipelines

## Legacy Reproduction Scripts

The original paper-focused scripts are preserved:
- `python FDCATE.py --help`
- `python analyze_fars_2000_fd.py --help`

## Development

```bash
python -m pip install -e .[dev]
python -m pytest -q
python -m build
```
