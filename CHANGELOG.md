# Changelog

## 0.1.0
- Add `fd_cate` standard-library package scaffold.
- Add sklearn-style `FDCATE` estimator wrapper (`fit/effect/summary`).
- Add `fdcate` CLI (`fit/effect/doctor/synthetic`).
- Add `fdcate benchmark` CLI for deterministic quick FD benchmark runs.
- Add artifact contract writers (`summary.txt`, `results.json`, `diagnostics.json`, `effects.csv`).
- Add model save/load compatibility policy (same major.minor).
- Add benchmark golden regression test (`tests/test_benchmark_golden.py`).
- Add benchmark golden reference artifact (`tests/benchmark_quick_reference.json`).
- Add live demo helper script (`scripts/run_demo_quick.sh`) and README results snapshot table.
- Add test suite and GitHub Actions CI/release workflows.
