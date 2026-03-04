"""Public API for fd-cate."""

from __future__ import annotations

from ._version import __version__
from .estimator import FDCATE
from .io import from_dataframe

__all__ = ["FDCATE", "from_dataframe", "__version__"]
