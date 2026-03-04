import numpy as np

from fd_cate import FDCATE
from FDCATE import simulate_fd_data_md


def test_seed_determinism_for_training_effects():
    data = simulate_fd_data_md(n=70, d=5, seed=5)

    est1 = FDCATE(method="fd-pi", nuisance_learner="xgb", random_state=123)
    est2 = FDCATE(method="fd-pi", nuisance_learner="xgb", random_state=123)

    est1.fit(data.C, data.Y, t=data.X, m=data.Z)
    est2.fit(data.C, data.Y, t=data.X, m=data.Z)

    assert np.allclose(est1.tau_train_, est2.tau_train_)
