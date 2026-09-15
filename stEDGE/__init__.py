#!/usr/bin/env python
"""
stEDGE: Edge-guided multiscale reconstruction of hierarchical spatial domains
and transition interfaces in spatial transcriptomics.
"""

__author__ = "Yi He"
__email__ = "yihe_csu@csu.edu.cn"
__version__ = "1.1.2"

import importlib
from .stEDGE import StEDGE

_SUBMODULES = {
    "utils",
    "preprocess",
    "plot",
    "TGL_module",
    "util_multi_scale",
    "stEDGE",
}

def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["StEDGE", *_SUBMODULES]