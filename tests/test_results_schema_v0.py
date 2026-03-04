import json

from FDCATE import simulate_fd_data_md

from fd_cate import FDCATE
from fd_cate.artifacts import write_artifacts


def test_results_schema_keys(tmp_path):
    data = simulate_fd_data_md(n=50, d=4, seed=9)
    est = FDCATE(method="fd-pi", nuisance_learner="xgb", random_state=9)
    est.fit(data.C, data.Y, t=data.X, m=data.Z)

    effects = est.effect(data.C)
    write_artifacts(
        outdir=tmp_path,
        estimator=est,
        effects=effects,
        input_spec={
            "n": int(data.C.shape[0]),
            "d": int(data.C.shape[1]),
            "treatment_type": "binary",
            "mediator_type": "binary",
            "outcome_type": "continuous_or_binary",
        },
        command_line="pytest",
        data_hash=None,
    )

    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert payload["schema_name"] == "fdcate.results"
    assert payload["schema_version"] == 0
    assert "provenance" in payload
    assert "estimator" in payload
    assert "outputs" in payload
