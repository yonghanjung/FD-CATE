"""Public API for fd-cate."""

from __future__ import annotations

from ._version import __version__
from .benchmark import run_multiseed_benchmark, run_quick_benchmark
from .demo import run_demo
from .estimator import FDCATE
from .io import from_dataframe

__all__ = [
    "FDCATE",
    "from_dataframe",
    "run_quick_benchmark",
    "run_multiseed_benchmark",
    "run_demo",
    "__version__",
]
