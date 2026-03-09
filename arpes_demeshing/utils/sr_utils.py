# -*- coding: UTF-8 -*-
# Super-resolution utilities: downscale, upscale, guided reconstruction
import cv2
import numpy as np


def downscale(img, target_size=(256, 256)):
    """Downscale a 2D image to the specified size.

    Uses INTER_AREA interpolation (optimal anti-aliasing for downscaling).

    Args:
        img: 2D numpy array (H, W)
        target_size: (target_H, target_W)

    Returns:
        Downscaled image (target_H, target_W)
    """
    # cv2.resize takes (width, height), need to swap
    return cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)


def upscale(img, target_size, method='lanczos'):
    """Upscale a 2D image to the specified size.

    Args:
        img: 2D numpy array (H, W)
        target_size: (target_H, target_W)
        method: 'lanczos' (highest quality) | 'cubic' | 'linear'

    Returns:
        Upscaled image (target_H, target_W)
    """
    interp = {
        'lanczos': cv2.INTER_LANCZOS4,
        'cubic': cv2.INTER_CUBIC,
        'linear': cv2.INTER_LINEAR,
    }.get(method, cv2.INTER_LANCZOS4)
    return cv2.resize(img, (target_size[1], target_size[0]), interpolation=interp)


def guided_upsample(texture_lr, original_shape):
    """Route 1: Pure Lanczos upscale (no residual compensation to avoid re-introducing mesh).

    Args:
        texture_lr: Low-resolution signal texture (H_lr, W_lr)
        original_shape: Original image size (H, W)

    Returns:
        Upscaled signal texture (H, W)
    """
    return upscale(texture_lr, original_shape)


def prepare_coarse_init(texture_1_lr, texture_2_lr, noise_1_lr, noise_2_lr, original_shape, input_depth):
    """Route 2: Upscale low-resolution results for coarse-to-fine Stage 2 initialization.

    Upscales network outputs to original resolution.

    Args:
        texture_1_lr: net_1 low-res output (1, 1, H_lr, W_lr)
        texture_2_lr: net_2 low-res output (1, 1, H_lr, W_lr)
        noise_1_lr: net_1 input noise (unused, reference only)
        noise_2_lr: net_2 input noise (unused, reference only)
        original_shape: Original image size (H, W)
        input_depth: Noise input channel count

    Returns:
        texture_1_hr, texture_2_hr: Upscaled texture references (H, W)
    """
    t1 = np.squeeze(texture_1_lr)
    t2 = np.squeeze(texture_2_lr)
    texture_1_hr = upscale(t1, original_shape)
    texture_2_hr = upscale(t2, original_shape)
    return texture_1_hr, texture_2_hr


def compute_psnr(img1, img2):
    """Compute PSNR (dB) between two images.

    Args:
        img1, img2: Same-sized numpy arrays, value range [0, 1]

    Returns:
        PSNR value (float)
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)
