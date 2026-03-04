#!/usr/bin/env bash
set -euo pipefail

# Quick live demo: synthetic fit/effect + quick benchmark report.
# Usage:
#   bash scripts/run_demo_quick.sh
# Optional env vars:
#   PYTHON_BIN=python3
#   OUT_BASE=/tmp/fdcate_live_demo

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_BASE="${OUT_BASE:-/tmp/fdcate_live_demo}"

echo "[demo] output directory: ${OUT_BASE}"
mkdir -p "${OUT_BASE}"

echo "[demo] 1) synthetic data"
PYTHONPATH=src "${PYTHON_BIN}" -m fd_cate synthetic \
  --n 120 --d 6 --seed 2026 \
  --out "${OUT_BASE}/synthetic.csv"

echo "[demo] 2) fit model + artifact contract"
PYTHONPATH=src "${PYTHON_BIN}" -m fd_cate fit \
  --data "${OUT_BASE}/synthetic.csv" \
  --outcome y --treat t --med m \
  --method fd-dr --nuisance-learner xgb \
  --outdir "${OUT_BASE}/fit_out"

echo "[demo] 3) effects from saved model"
PYTHONPATH=src "${PYTHON_BIN}" -m fd_cate effect \
  --model "${OUT_BASE}/fit_out/model.pkl" \
  --data "${OUT_BASE}/synthetic.csv" \
  --covariates x0,x1,x2,x3,x4,x5 \
  --out "${OUT_BASE}/effects_from_model.csv"

echo "[demo] 4) quick benchmark"
PYTHONPATH=src "${PYTHON_BIN}" -m fd_cate benchmark \
  --n 60 --d 4 --seed 17 \
  --nuisance-learner xgb \
  --out "${OUT_BASE}/benchmark_quick.json"

echo "[demo] done"
echo "  - ${OUT_BASE}/fit_out/summary.txt"
echo "  - ${OUT_BASE}/fit_out/results.json"
echo "  - ${OUT_BASE}/fit_out/diagnostics.json"
echo "  - ${OUT_BASE}/fit_out/effects.csv"
echo "  - ${OUT_BASE}/fit_out/model.pkl"
echo "  - ${OUT_BASE}/benchmark_quick.json"
