import math
import json
from pathlib import Path

from fd_cate.benchmark import run_quick_benchmark


GOLDEN_PATH = Path(__file__).with_name("benchmark_quick_reference.json")


def test_benchmark_quick_golden_snapshot():
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    cfg = golden["config"]
    expected = golden["results"]
    report = run_quick_benchmark(
        n=int(cfg["n"]),
        d=int(cfg["d"]),
        seed=int(cfg["seed"]),
        learner=str(cfg["learner"]),
    )

    assert report["schema_name"] == "fdcate.benchmark"
    assert report["schema_version"] == 0
    assert report["config"]["profile"] == "quick"
    assert report["config"]["scenarios"] == ["clean", "weak-overlap"]

    for scenario, expected_metrics in expected.items():
        observed_metrics = report["results"][scenario]
        for metric_name, expected_value in expected_metrics.items():
            observed = float(observed_metrics[metric_name])
            assert math.isfinite(observed)
            # Keep this tolerant to absorb small numerical drift across versions/platforms.
            assert abs(observed - expected_value) <= 0.20
