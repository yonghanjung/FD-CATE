import numpy as np
import pytest
import joblib

from fd_cate import FDCATE
from FDCATE import simulate_fd_data_md


def test_fit_and_effect_smoke():
    data = simulate_fd_data_md(n=80, d=5, seed=11)
    est = FDCATE(method="fd-dr", nuisance_learner="xgb", random_state=3)
    est.fit(data.C, data.Y, t=data.X, m=data.Z)

    tau = est.effect(data.C[:10])
    assert tau.shape == (10,)
    assert np.all(np.isfinite(tau))
    assert np.isfinite(est.ate_)


def test_fd_r_config_smoke():
    data = simulate_fd_data_md(n=60, d=4, seed=12)
    est = FDCATE(
        method="fd-r",
        nuisance_learner="xgb",
        fd_r_b_learner="xgb",
        fd_r_g_solver="ratio",
        fd_r_swap_average=False,
        random_state=4,
    )
    est.fit(data.C, data.Y, t=data.X, m=data.Z)

    tau = est.effect(data.C[:8])
    assert tau.shape == (8,)
    assert np.all(np.isfinite(tau))


def test_model_compatibility_major_minor_guard(tmp_path):
    data = simulate_fd_data_md(n=40, d=4, seed=13)
    est = FDCATE(method="fd-pi", nuisance_learner="xgb", random_state=5)
    est.fit(data.C, data.Y, t=data.X, m=data.Z)

    model_path = tmp_path / "model.pkl"
    est.save(model_path)

    payload = joblib.load(model_path)
    payload["meta"]["package_version"] = "0.2.0"
    joblib.dump(payload, model_path)

    with pytest.raises(RuntimeError, match="Compatibility policy: same major.minor only."):
        FDCATE.load(model_path)
