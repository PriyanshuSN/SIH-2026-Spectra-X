"""
mc_dropout.py — Monte Carlo Dropout uncertainty estimation.

How it works:
  1. Keep dropout layers ACTIVE during inference (model.train() for dropout only)
  2. Run N stochastic forward passes on the same input
  3. Compute per-pixel mean (= final SR output) and variance (= uncertainty)
  4. Normalize variance to [0, 1] → uncertainty heatmap

High variance = model is uncertain about that pixel → flag it.
"""

import numpy as np
import torch


def enable_mc_dropout(model: torch.nn.Module):
    """
    Enable dropout layers while keeping everything else in eval mode.

    This is the key trick: we want BatchNorm/LayerNorm in eval mode
    (use running stats) but Dropout in train mode (randomly drop units).
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    n_passes: int = 8,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run MC-Dropout inference to get SR output + uncertainty map.

    Args:
        model: Trained SwinIR-Lite model.
        input_tensor: Input of shape (1, 4, H, W) — single tile, 4 bands.
        n_passes: Number of stochastic forward passes (more = better uncertainty,
                  but slower). Default 8 for demo, use 16+ for production.
        device: 'cuda' or 'cpu'.

    Returns:
        Tuple of:
            - mean_output: np.ndarray of shape (4, H*scale, W*scale) — averaged SR result
            - uncertainty: np.ndarray of shape (H*scale, W*scale) — normalized [0, 1]
              uncertainty heatmap (averaged across bands)
    """
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Enable MC-Dropout
    enable_mc_dropout(model)

    # Collect N stochastic predictions
    predictions = []
    with torch.no_grad():
        for _ in range(n_passes):
            output = model(input_tensor)  # (1, 4, H*s, W*s)
            predictions.append(output.cpu().numpy())

    # Stack: (N, 1, 4, H*s, W*s)
    predictions = np.stack(predictions, axis=0)

    # Mean across passes → final SR output
    mean_output = predictions.mean(axis=0).squeeze(0)  # (4, H*s, W*s)

    # Variance across passes → uncertainty
    variance = predictions.var(axis=0).squeeze(0)  # (4, H*s, W*s)

    # Average variance across bands → single heatmap
    uncertainty = variance.mean(axis=0)  # (H*s, W*s)

    # Normalize to [0, 1]
    u_min, u_max = uncertainty.min(), uncertainty.max()
    if u_max > u_min:
        uncertainty = (uncertainty - u_min) / (u_max - u_min)
    else:
        uncertainty = np.zeros_like(uncertainty)

    return mean_output, uncertainty
