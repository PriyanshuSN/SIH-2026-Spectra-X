"""
compute.py — Quality metrics for satellite super-resolution evaluation.

Implements four standard remote-sensing quality metrics:
  - PSNR: Peak Signal-to-Noise Ratio (higher is better)
  - SSIM: Structural Similarity Index (higher is better, max 1.0)
  - SAM: Spectral Angle Mapper (lower is better, in degrees)
  - ERGAS: Erreur Relative Globale Adimensionnelle de Synthèse (lower is better)

SAM and ERGAS are critical for multi-band satellite imagery —
they measure spectral fidelity, not just visual similarity.
"""

import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio as _psnr,
)
from skimage.metrics import (
    structural_similarity as _ssim,
)


def compute_psnr(reference: np.ndarray, target: np.ndarray) -> float:
    """
    Compute PSNR averaged across all bands.

    Args:
        reference: Ground truth, shape (C, H, W), values in [0, 1].
        target: SR output, shape (C, H, W), values in [0, 1].

    Returns:
        PSNR in dB (higher is better).
    """
    return float(_psnr(reference, target, data_range=1.0))


def compute_ssim(reference: np.ndarray, target: np.ndarray) -> float:
    """
    Compute SSIM averaged across bands.

    Args:
        reference: Ground truth, shape (C, H, W).
        target: SR output, shape (C, H, W).

    Returns:
        SSIM value in [0, 1] (higher is better).
    """
    C = reference.shape[0]
    ssim_values = []
    for c in range(C):
        val = _ssim(reference[c], target[c], data_range=1.0)
        ssim_values.append(val)
    return float(np.mean(ssim_values))


def compute_sam(reference: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Spectral Angle Mapper (SAM).

    SAM measures the spectral fidelity by computing the angle between
    the reference and target spectral vectors at each pixel, then averaging.

    Args:
        reference: Ground truth, shape (C, H, W).
        target: SR output, shape (C, H, W).

    Returns:
        SAM in degrees (lower is better). Typical good values: < 5°.
    """
    # Reshape to (C, N) where N = H*W
    C = reference.shape[0]
    ref_flat = reference.reshape(C, -1)  # (C, N)
    tgt_flat = target.reshape(C, -1)  # (C, N)

    # Dot product per pixel
    dot = np.sum(ref_flat * tgt_flat, axis=0)  # (N,)

    # Norms per pixel
    ref_norm = np.sqrt(np.sum(ref_flat ** 2, axis=0) + 1e-10)
    tgt_norm = np.sqrt(np.sum(tgt_flat ** 2, axis=0) + 1e-10)

    # Cosine similarity, clamp to [-1, 1]
    cos_sim = np.clip(dot / (ref_norm * tgt_norm), -1.0, 1.0)

    # Angle in degrees
    angles = np.degrees(np.arccos(cos_sim))

    return float(np.mean(angles))


def compute_ergas(
    reference: np.ndarray,
    target: np.ndarray,
    scale_factor: int = 3,
) -> float:
    """
    Compute ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse).

    ERGAS is a global multi-band relative error metric commonly used
    in remote sensing to evaluate pan-sharpening and super-resolution.

    Args:
        reference: Ground truth, shape (C, H, W).
        target: SR output, shape (C, H, W).
        scale_factor: Resolution ratio (e.g., 3 for 10m → ~3.3m).

    Returns:
        ERGAS value (lower is better). Typical good values: < 3.
    """
    C = reference.shape[0]
    ratio = 100.0 / scale_factor

    band_errors = []
    for c in range(C):
        ref_band = reference[c]
        tgt_band = target[c]
        mean_ref = np.mean(ref_band)

        if mean_ref < 1e-10:
            continue

        rmse = np.sqrt(np.mean((ref_band - tgt_band) ** 2))
        band_errors.append((rmse / mean_ref) ** 2)

    if len(band_errors) == 0:
        return 0.0

    ergas = ratio * np.sqrt(np.mean(band_errors))
    return float(ergas)


def compute_all_metrics(
    reference: np.ndarray,
    target: np.ndarray,
    scale_factor: int = 3,
) -> dict:
    """
    Compute all four quality metrics at once.

    Args:
        reference: Ground truth, shape (C, H, W), values in [0, 1].
        target: SR output, shape (C, H, W), values in [0, 1].

    Returns:
        Dict with keys: 'psnr', 'ssim', 'sam', 'ergas'.
    """
    return {
        "psnr": compute_psnr(reference, target),
        "ssim": compute_ssim(reference, target),
        "sam": compute_sam(reference, target),
        "ergas": compute_ergas(reference, target, scale_factor),
    }
