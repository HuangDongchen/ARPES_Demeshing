# -*- coding: UTF-8 -*-
"""ARPES Demeshing — Dual U-Net Deep Image Prior for mesh artifact removal.

Quick start::

    import arpes_demeshing as ad

    data, header = ad.load_txt("spectrum.txt")
    result = ad.demesh(data, mode='fast', num_iter=3000)

    # result.signal  → clean spectrum (np.ndarray)
    # result.mesh    → extracted mesh pattern
"""
from ._config import DemeshConfig, DemeshResult
from ._device import get_best_device
from ._mask import make_rect_mask
from .core import demesh
from .io import load_ibw, load_pxt, load_txt, plot_comparison, save_result, save_result_ibw

__version__ = "0.2.0"

__all__ = [
    "demesh",
    "get_best_device",
    "DemeshConfig",
    "DemeshResult",
    "load_txt",
    "load_pxt",
    "load_ibw",
    "save_result",
    "save_result_ibw",
    "plot_comparison",
    "make_rect_mask",
]

