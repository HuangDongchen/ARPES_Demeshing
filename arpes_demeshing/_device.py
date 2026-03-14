# -*- coding: UTF-8 -*-
"""Automatic device detection for ARPES demeshing.

Priority: CUDA → MPS (Apple Silicon) → CPU.
"""
import torch


def get_best_device():
    """Return the best available torch device.

    Detection order:
    1. NVIDIA CUDA  (``torch.cuda.is_available()``)
    2. Apple MPS    (``torch.backends.mps.is_available()``, macOS 12.3+)
    3. CPU fallback

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        print(f"[arpes_demeshing] Using CUDA device: {name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[arpes_demeshing] Using Apple MPS (Metal Performance Shaders)")
    else:
        dev = torch.device("cpu")
        print("[arpes_demeshing] Using CPU (no GPU detected)")
    return dev
