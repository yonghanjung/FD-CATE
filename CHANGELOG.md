# Changelog

## 0.1.1
- Remove Korean wording from Quickstart heading for package-index rendering consistency.
- Fix benchmark figure links to absolute GitHub raw URLs so images render on package index pages.

## 0.1.0
- Add `fd_cate` standard-library package scaffold.
- Add sklearn-style `FDCATE` estimator wrapper (`fit/effect/summary`).
- Add `fdcate` CLI (`fit/effect/doctor/synthetic`).
- Add one-click `fdcate demo` command (synthetic -> fit artifacts -> optional benchmark).
- Add `fdcate benchmark` CLI for deterministic quick FD benchmark runs.
- Add multiseed benchmark profile (`--profile multiseed`) with summary stats.
- Expose FD-R benchmark/fit controls (`fd_r_b_learner`, `fd_r_g_solver`, `fd_r_swap_average`).
- Add artifact contract writers (`summary.txt`, `results.json`, `diagnostics.json`, `effects.csv`).
- Add model save/load compatibility policy (same major.minor).
- Add benchmark golden regression test (`tests/test_benchmark_golden.py`).
- Add benchmark profile contract tests (`tests/test_benchmark_profiles.py`).
- Add benchmark golden reference artifact (`tests/benchmark_quick_reference.json`).
- Add expanded demo contract tests (defaults, overwrite determinism, `nn`, `fd-r`, invalid bool).
- Add diagnostics artifact schema wrapper (`fdcate.diagnostics`, `schema_version=0`).
- Add nightly/manual slow-test workflow (`.github/workflows/slow-tests.yml`).
- Add live demo helper script (`scripts/run_demo_quick.sh`) and README results snapshot table.
- Add release preflight script (`scripts/release_preflight.sh`) and runbook (`RELEASE_RUNBOOK.md`).
- Add test suite and GitHub Actions CI/release workflows.
