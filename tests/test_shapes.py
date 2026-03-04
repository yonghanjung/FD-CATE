import numpy as np

from fd_cate import FDCATE
from FDCATE import simulate_fd_data_md


def test_effect_direction_scaling():
    data = simulate_fd_data_md(n=60, d=4, seed=7)
    est = FDCATE(method="fd-pi", nuisance_learner="xgb", random_state=7)
    est.fit(data.C, data.Y, t=data.X, m=data.Z)

    tau01 = est.effect(data.C[:12], t0=0, t1=1)
    tau10 = est.effect(data.C[:12], t0=1, t1=0)
    assert tau01.shape == (12,)
    assert np.allclose(tau01, -tau10)
