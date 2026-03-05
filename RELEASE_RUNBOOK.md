# FD-CATE v0.1.0 Release Runbook

This runbook is the exact release checklist for `v0.1.0`.

## 0. Release lane checks

```bash
git branch --show-current
git status --short
```

Expected:
- Branch is `release/v0.1.0-pypi`
- Working tree is clean

## 1. Local preflight (fast gate)

```bash
bash scripts/release_preflight.sh
```

This runs:
- `pytest -m "not slow"`
- `python -m build`
- fresh-venv wheel smoke
- `fdcate demo --run-benchmark false` artifact checks

## 2. Version and changelog freeze

Confirm:
- `pyproject.toml` has `version = "0.1.0"`
- `CHANGELOG.md` has frozen `0.1.0` section

## 3. Tag and push

```bash
git tag -a v0.1.0 -m "v0.1.0"
git push origin release/v0.1.0-pypi
git push origin v0.1.0
```

## 4. GitHub Actions gate verification

Workflow: `.github/workflows/release.yml`

Required order and gates:
1. `build-dist`
2. `publish-testpypi`
3. `install-smoke-testpypi`
4. `publish-pypi` (must wait for step 3 success)

If `install-smoke-testpypi` fails, `publish-pypi` must not run.

## 5. Post-release fresh-venv verification

```bash
python3 -m venv /tmp/fdcate-pypi-verify
/tmp/fdcate-pypi-verify/bin/python -m pip install -U pip
/tmp/fdcate-pypi-verify/bin/python -m pip install fd-cate
/tmp/fdcate-pypi-verify/bin/fdcate demo --outdir /tmp/fdcate-pypi-smoke --run-benchmark false
```

Required files:
- `/tmp/fdcate-pypi-smoke/synthetic.csv`
- `/tmp/fdcate-pypi-smoke/fit_out/model.pkl`
- `/tmp/fdcate-pypi-smoke/fit_out/results.json`
- `/tmp/fdcate-pypi-smoke/fit_out/diagnostics.json`
- `/tmp/fdcate-pypi-smoke/fit_out/effects.csv`

## 6. Rollback notes

- If TestPyPI publish fails: fix packaging metadata and retag with next patch version.
- If TestPyPI install smoke fails: do not publish to PyPI; fix and retag.
- If PyPI publish succeeds but smoke fails: publish immediate patch (`v0.1.1`) with fixed installer/runtime path.
