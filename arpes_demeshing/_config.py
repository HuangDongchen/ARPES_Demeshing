# -*- coding: UTF-8 -*-
"""Configuration and result dataclasses for ARPES demeshing."""
from dataclasses import dataclass

import numpy as np


@dataclass
class DemeshConfig:
    """All hyperparameters for the demeshing pipeline."""

    target_size: int = 256
    mode: str = 'fast'          # 'fast' | 'quality'
    num_iter: int = 2000
    num_iter_fine: int = -1     # auto: num_iter // 2
    use_coord: bool = True
    coord_mode: str = 'signal_only'  # 'signal_only' | 'both'
    mesh_l1: float = 0.05
    width: int = 128
    num_scales: int = 5
    skip_channels: int = 4
    lr_signal: float = 0.5
    lr_mesh: float = 1.0
    optimizer: str = 'sgd'      # 'sgd' | 'adam'
    lambda_max: float = 0.5


@dataclass
class DemeshResult:
    """Result container returned by ``demesh()``."""

    signal: np.ndarray          # (H, W) denormalized clean signal
    mesh: np.ndarray            # (H, W) denormalized mesh pattern
    signal_norm: np.ndarray     # (H, W) normalized [0, 1] signal
    mesh_norm: np.ndarray       # (H, W) normalized [0, 1] mesh
    loss_history: list          # training loss records
    config: DemeshConfig
    norm_const: float           # normalization constant
