# -*- coding: UTF-8 -*-
"""Mask utilities for excluding regions from loss computation."""
import numpy as np


def make_rect_mask(shape, x, y, w, h):
    """Create a rectangular boolean mask.

    Pixels inside the rectangle are set to 1.0 (excluded from loss).

    Args:
        shape: (H, W) tuple — image dimensions.
        x: Top-left column index.
        y: Top-left row index.
        w: Rectangle width (pixels).
        h: Rectangle height (pixels).

    Returns:
        np.ndarray of shape (H, W), float32, with 1.0 inside the rectangle.
    """
    mask = np.zeros(shape, dtype=np.float32)
    y0 = max(0, int(y))
    x0 = max(0, int(x))
    y1 = min(shape[0], y0 + int(h))
    x1 = min(shape[1], x0 + int(w))
    if y1 > y0 and x1 > x0:
        mask[y0:y1, x0:x1] = 1.0
    return mask


def masked_mse_loss(pred, target, mask, eps=1e-8):
    """MSE loss computed only on the unmasked region (complement of *mask*).

    Args:
        pred: Predicted tensor, broadcastable to (B, C, H, W).
        target: Target tensor, same shape as *pred*.
        mask: Binary mask tensor — 1.0 = excluded, 0.0 = included.
        eps: Clamping value to avoid division by zero.

    Returns:
        Scalar loss tensor.
    """
    pred = pred.float()
    target = target.float()
    mask = mask.to(dtype=pred.dtype)

    mask_comp = 1.0 - mask
    diff2 = (pred - target) ** 2
    numer = (diff2 * mask_comp).sum()
    denom = mask_comp.sum().clamp_min(eps)
    return numer / denom
