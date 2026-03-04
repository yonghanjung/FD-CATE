import numpy as np

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
