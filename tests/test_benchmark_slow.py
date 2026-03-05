import math

import pytest

from fd_cate.benchmark import run_multiseed_benchmark


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_multiseed_profile_slow_smoke():
    report = run_multiseed_benchmark(
        n=120,
        d=8,
        seed=2026,
        n_seeds=5,
        learner="xgb",
        fd_r_g_solver="direct",
        fd_r_b_learner="xgb",
    )
    agg = report["results"]["summary"]["aggregate_mean_rmse"]
    for method in ("fd-pi", "fd-dr", "fd-r"):
        assert math.isfinite(float(agg[method]["mean"]))
