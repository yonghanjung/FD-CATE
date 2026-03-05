#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TMP_VENV="${TMP_VENV:-/tmp/fdcate-v010-preflight-venv}"
TMP_OUTDIR="${TMP_OUTDIR:-/tmp/fdcate-v010-preflight-demo}"

echo "[preflight] repo: ${ROOT_DIR}"
echo "[preflight] python: $(python3 --version)"
echo "[preflight] running fast tests"
python3 -m pytest -q -m "not slow"

echo "[preflight] building wheel/sdist"
python3 -m build

echo "[preflight] wheel install smoke"
rm -rf "${TMP_VENV}" "${TMP_OUTDIR}"
python3 -m venv "${TMP_VENV}"
"${TMP_VENV}/bin/python" -m pip install -U pip
"${TMP_VENV}/bin/python" -m pip install dist/*.whl
"${TMP_VENV}/bin/fdcate" --help >/dev/null
"${TMP_VENV}/bin/fdcate" demo --outdir "${TMP_OUTDIR}" --run-benchmark false --n 40 --d 4 --seed 0 >/dev/null

test -f "${TMP_OUTDIR}/synthetic.csv"
test -f "${TMP_OUTDIR}/fit_out/model.pkl"
test -f "${TMP_OUTDIR}/fit_out/results.json"
test -f "${TMP_OUTDIR}/fit_out/diagnostics.json"
test -f "${TMP_OUTDIR}/fit_out/effects.csv"

echo "[preflight] OK"
