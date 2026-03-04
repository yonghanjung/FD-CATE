# FD-CATE v0.1.0 (Draft)

Release date: Draft (2026-03-04)

## Summary

`v0.1.0` introduces the first standard-library interface for FD-CATE while preserving paper-parity defaults from the legacy scripts.

## Highlights

- Added installable package scaffold (`pyproject.toml`, `src/fd_cate`, tests, CI, release workflow).
- Added sklearn-style estimator API:
  - `from fd_cate import FDCATE`
  - `fit(X, y, t=..., m=...)`
  - `effect(X_new, t0=0, t1=1)`
- Added CLI:
  - `fdcate synthetic`
  - `fdcate fit`
  - `fdcate effect`
  - `fdcate doctor`
- Added fixed artifact contract for `fit`:
  - `summary.txt`, `results.json`, `diagnostics.json`, `effects.csv`, `model.pkl`
- Added diagnostics guardrails (binary checks, overlap proxy, first-stage and nuisance quality warnings).

## Dependency and Learner Policy (v0.1)

- Default learner remains **XGBoost** (`nuisance_learner="xgb"`).
- `nn` learner is supported (`nuisance_learner="nn"`).
- `xgboost` is included in core dependencies to keep default path (`pip install fd-cate`) runnable.

## Model Compatibility Policy

`model.pkl` load is allowed only when package **major.minor** versions match.

- Allowed: `0.1.x -> 0.1.y`
- Blocked: `0.1.x -> 0.2.x`

## Legacy Script Status

Legacy research scripts are preserved and packaged:

- `FDCATE.py`
- `analyze_fars_2000_fd.py`

## Validation Snapshot

Commands used in local validation:

```bash
python -m pytest -q
python -m build
fdcate --help
fdcate synthetic --n 40 --d 4 --seed 0 --out /tmp/fdcate_synth.csv
fdcate fit --data /tmp/fdcate_synth.csv --outcome y --treat t --med m --outdir /tmp/fdcate_out --method fd-pi --nuisance-learner xgb
fdcate effect --model /tmp/fdcate_out/model.pkl --data /tmp/fdcate_synth.csv --covariates x0,x1,x2,x3 --out /tmp/fdcate_out/effects_reload.csv
fdcate doctor --data /tmp/fdcate_synth.csv --outcome y --treat t --med m --covariates x0,x1,x2,x3
```

## Known Limits (v0.1)

- Supports binary `T` and binary `M` only.
- No auto categorical feature pipeline.
- CI smoke currently runs a synthetic fit path; large experimental pipelines remain outside CI.
