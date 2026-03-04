"""Public API for fd-cate."""

from __future__ import annotations

from ._version import __version__
from .benchmark import run_quick_benchmark
from .estimator import FDCATE
from .io import from_dataframe

__all__ = ["FDCATE", "from_dataframe", "run_quick_benchmark", "__version__"]
