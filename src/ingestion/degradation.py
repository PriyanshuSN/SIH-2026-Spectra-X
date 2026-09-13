"""
degradation.py — Self-supervised degradation pipeline.

Creates training pairs by synthetically downsampling Sentinel-2 tiles:
  - Original 10m tile = "high-resolution" target
  - Degraded version (bicubic + blur + noise) = "low-resolution" input
  - Model learns to reverse the degradation

This is the SAME degradation kernel used in the consistency check (src/consistency/).
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def degrade(
    image: np.ndarray,
    scale_factor: int = 3,
    blur_sigma: float = 1.5,
    noise_std: float = 0.01,
) -> np.ndarray:
    """
    Apply synthetic degradation to simulate a lower-resolution input.

    Pipeline: Gaussian blur → bicubic downsample → add Gaussian noise

    Args:
        image: Input array of shape (C, H, W), values in [0, 1].
        scale_factor: Downsampling factor (e.g., 3 means 10m → ~30m equivalent).
        blur_sigma: Sigma for Gaussian blur (anti-aliasing).
        noise_std: Standard deviation of additive Gaussian noise.

    Returns:
        Degraded array of shape (C, H//scale_factor, W//scale_factor).
    """
    C, H, W = image.shape

    # Step 1: Gaussian blur (per-band)
    blurred = np.zeros_like(image)
    for c in range(C):
        blurred[c] = gaussian_filter(image[c], sigma=blur_sigma)

    # Step 2: Bicubic downsample
    new_H, new_W = H // scale_factor, W // scale_factor
    # Use simple area-averaging as a NumPy-native approach
    downsampled = _area_downsample(blurred, new_H, new_W)

    # Step 3: Add noise
    noise = np.random.normal(0, noise_std, downsampled.shape).astype(np.float32)
    degraded = np.clip(downsampled + noise, 0, 1)

    return degraded


def _area_downsample(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Area-averaging downsample (simple, reliable, no PIL dependency)."""
    C, H, W = image.shape
    # Crop to be divisible
    crop_h = (H // target_h) * target_h
    crop_w = (W // target_w) * target_w
    cropped = image[:, :crop_h, :crop_w]

    # Reshape and average
    scale_h = crop_h // target_h
    scale_w = crop_w // target_w
    reshaped = cropped.reshape(C, target_h, scale_h, target_w, scale_w)
    return reshaped.mean(axis=(2, 4)).astype(np.float32)


def create_training_pair(
    image: np.ndarray,
    scale_factor: int = 3,
    blur_sigma: float = 1.5,
    noise_std: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a (low_res, high_res) training pair from a single image.

    Args:
        image: Original tile of shape (C, H, W), values in [0, 1].
        scale_factor: Downsampling factor.

    Returns:
        Tuple of (degraded_input, original_target).
    """
    lr = degrade(image, scale_factor, blur_sigma, noise_std)
    # Crop HR to be compatible with the degraded size
    C, H, W = image.shape
    new_H, new_W = H // scale_factor * scale_factor, W // scale_factor * scale_factor
    hr = image[:, :new_H, :new_W]

    return lr, hr
