"""
inference.py — Unified inference pipeline.

Single entry point that orchestrates:
  1. Preprocessing (normalize, tile)
  2. SR model forward pass
  3. MC-Dropout uncertainty estimation
  4. Consistency check
  5. Quality metrics (if reference is available)

This is the function called by both the Streamlit UI and future FastAPI backend.
The ML core is UI-agnostic by design — swap the frontend without touching this.
"""

import numpy as np
import torch

from src.model.swinir import build_model
from src.ingestion.preprocess import normalize
from src.ingestion.tiler import tile_image, untile_image
from src.uncertainty.mc_dropout import mc_dropout_inference
from src.consistency.checker import consistency_check
from src.metrics.compute import compute_all_metrics


def load_model(
    checkpoint_path: str,
    model_config: dict = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Load a trained SwinIR-Lite model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        model_config: Optional model config overrides.
        device: 'cuda' or 'cpu'.

    Returns:
        Loaded model in eval mode.
    """
    model = build_model(model_config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def run_inference(
    model: torch.nn.Module,
    input_image: np.ndarray,
    reference_image: np.ndarray = None,
    n_uncertainty_passes: int = 8,
    scale_factor: int = 3,
    patch_size: int = 256,
    overlap: int = 16,
    device: str = "cpu",
) -> dict:
    """
    Run the complete SpectraX inference pipeline.

    Args:
        model: Trained SwinIR-Lite model.
        input_image: LR input of shape (4, H, W), raw pixel values.
        reference_image: Optional HR reference of shape (4, H*scale, W*scale)
                         for computing quality metrics. None if unavailable.
        n_uncertainty_passes: Number of MC-Dropout stochastic passes.
        scale_factor: Model upscaling factor.
        patch_size: Patch size for tiling.
        overlap: Overlap between patches.
        device: 'cuda' or 'cpu'.

    Returns:
        Dict with:
            - 'sr_output': np.ndarray (4, H*scale, W*scale) — super-resolved image
            - 'uncertainty_map': np.ndarray (H*scale, W*scale) — [0,1] confidence heatmap
            - 'consistency': dict — consistency check results
            - 'metrics': dict or None — PSNR/SSIM/SAM/ERGAS (only if reference given)
    """
    # Normalize input
    normalized = normalize(input_image.copy(), method="percentile")

    # Tile for memory-safe inference
    patches = tile_image(normalized, patch_size=patch_size, overlap=overlap)

    sr_patches = []
    uncertainty_patches = []

    for p in patches:
        patch_tensor = torch.from_numpy(p["patch"]).unsqueeze(0).float()  # (1, 4, H, W)

        # MC-Dropout inference (get SR + uncertainty for each patch)
        sr_patch, unc_patch = mc_dropout_inference(
            model, patch_tensor, n_passes=n_uncertainty_passes, device=device
        )

        sr_patches.append({**p, "patch": sr_patch})
        uncertainty_patches.append({**p, "patch": unc_patch[np.newaxis, ...]})  # (1, H, W)

    # Reassemble patches
    sr_output = untile_image(
        sr_patches,
        normalized.shape,
        patch_size=patch_size,
        scale_factor=scale_factor,
    )

    uncertainty_map = untile_image(
        uncertainty_patches,
        (1, normalized.shape[1], normalized.shape[2]),
        patch_size=patch_size,
        scale_factor=scale_factor,
    ).squeeze(0)  # (H*scale, W*scale)

    # Consistency check
    consistency = consistency_check(
        sr_output, normalized, scale_factor=scale_factor
    )

    # Quality metrics (only if reference is available)
    metrics = None
    if reference_image is not None:
        ref_normalized = normalize(reference_image.copy(), method="percentile")
        # Crop to matching size
        _, rH, rW = ref_normalized.shape
        _, sH, sW = sr_output.shape
        min_H, min_W = min(rH, sH), min(rW, sW)
        metrics = compute_all_metrics(
            ref_normalized[:, :min_H, :min_W],
            sr_output[:, :min_H, :min_W],
            scale_factor=scale_factor,
        )

    return {
        "sr_output": sr_output,
        "uncertainty_map": uncertainty_map,
        "consistency": consistency,
        "metrics": metrics,
    }
