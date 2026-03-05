import math

from fd_cate.benchmark import run_multiseed_benchmark, run_quick_benchmark


def test_multiseed_profile_schema_and_summary():
    report = run_multiseed_benchmark(
        n=40,
        d=4,
        seed=101,
        n_seeds=3,
        learner="xgb",
    )
    assert report["schema_name"] == "fdcate.benchmark"
    assert report["schema_version"] == 0
    assert report["config"]["profile"] == "multiseed"
    assert report["config"]["n_seeds"] == 3
    assert len(report["results"]["per_seed"]) == 3

    for method in ("fd-pi", "fd-dr", "fd-r"):
        stats = report["results"]["summary"]["aggregate_mean_rmse"][method]
        assert math.isfinite(float(stats["mean"]))
        assert float(stats["min"]) <= float(stats["max"])


def test_fd_r_solver_and_b_learner_are_exposed():
    report = run_quick_benchmark(
        n=30,
        d=3,
        seed=8,
        learner="xgb",
        fd_r_g_solver="ratio",
        fd_r_swap_average=False,
        fd_r_b_learner="xgb",
    )
    cfg = report["config"]["fd_r"]
    assert cfg["g_solver"] == "ratio"
    assert cfg["swap_average"] is False
    assert cfg["b_learner"] == "xgb"

