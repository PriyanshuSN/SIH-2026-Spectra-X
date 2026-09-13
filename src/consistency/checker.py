"""
checker.py — Self-consistency verification.

The key idea:
  If the SR output is physically consistent, then degrading it back
  to the original resolution should closely match the original input.

  SR_output --[degrade]--> degraded_SR
  Compare: degraded_SR vs. original_input
  → If they match → model output is consistent with observations
  → If they don't → model may have hallucinated detail

Uses the SAME degradation kernel from src/ingestion/degradation.py
to ensure the check is aligned with the training assumption.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim

from src.ingestion.degradation import degrade


def consistency_check(
    sr_output: np.ndarray,
    original_input: np.ndarray,
    scale_factor: int = 3,
    blur_sigma: float = 1.5,
    pass_threshold: float = 0.85,
) -> dict:
    """
    Run the self-consistency check on a super-resolved output.

    Args:
        sr_output: SR model output, shape (4, H*scale, W*scale), values in [0, 1].
        original_input: Original LR input, shape (4, H, W), values in [0, 1].
        scale_factor: Downsampling factor used in training/degradation.
        blur_sigma: Gaussian blur sigma (must match training degradation).
        pass_threshold: SSIM threshold for pass/fail (default 0.85).

    Returns:
        Dict with:
            - 'consistency_score': float (average SSIM across bands)
            - 'passed': bool (score >= threshold)
            - 'per_band_ssim': list of per-band SSIM values
            - 'pixel_error': float (mean absolute pixel error)
            - 'error_map': np.ndarray of shape (H, W) — spatial error distribution
    """
    # Degrade SR output back to original resolution (no noise, we want deterministic)
    degraded_sr = degrade(sr_output, scale_factor=scale_factor, blur_sigma=blur_sigma, noise_std=0.0)

    # Ensure shapes match
    C, H, W = original_input.shape
    degraded_sr = degraded_sr[:, :H, :W]

    # Per-band SSIM
    per_band_ssim = []
    for c in range(C):
        band_ssim = ssim(
            original_input[c],
            degraded_sr[c],
            data_range=1.0,
        )
        per_band_ssim.append(float(band_ssim))

    consistency_score = float(np.mean(per_band_ssim))

    # Pixel-level error
    error = np.abs(original_input - degraded_sr)
    pixel_error = float(error.mean())

    # Spatial error map (averaged across bands)
    error_map = error.mean(axis=0)  # (H, W)

    return {
        "consistency_score": consistency_score,
        "passed": consistency_score >= pass_threshold,
        "per_band_ssim": per_band_ssim,
        "pixel_error": pixel_error,
        "error_map": error_map,
    }
