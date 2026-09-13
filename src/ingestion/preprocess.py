"""
preprocess.py — Read GeoTIFF, extract and align B/G/R/NIR bands.

Usage:
    from src.ingestion.preprocess import load_sentinel2_bands
    bands = load_sentinel2_bands("path/to/tile.tif")
"""

import numpy as np


def load_sentinel2_bands(filepath: str, bands: list[int] = None) -> np.ndarray:
    """
    Load Sentinel-2 GeoTIFF and extract B/G/R/NIR bands.

    Args:
        filepath: Path to the GeoTIFF file.
        bands: List of band indices to extract (default: [2, 3, 4, 8] = B/G/R/NIR).

    Returns:
        numpy array of shape (4, H, W) — 4 bands at 10m resolution.
    """
    import rasterio

    if bands is None:
        bands = [2, 3, 4, 8]  # Blue, Green, Red, NIR for Sentinel-2

    with rasterio.open(filepath) as src:
        # Read specified bands (rasterio is 1-indexed)
        data = src.read(bands)
        profile = src.profile

    return data.astype(np.float32), profile


def normalize(data: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize pixel values to [0, 1] range.

    Args:
        data: Input array of shape (C, H, W).
        method: Normalization method — 'minmax' or 'percentile'.

    Returns:
        Normalized array of same shape.
    """
    if method == "minmax":
        dmin = data.min(axis=(1, 2), keepdims=True)
        dmax = data.max(axis=(1, 2), keepdims=True)
        return (data - dmin) / (dmax - dmin + 1e-8)
    elif method == "percentile":
        p2 = np.percentile(data, 2, axis=(1, 2), keepdims=True)
        p98 = np.percentile(data, 98, axis=(1, 2), keepdims=True)
        return np.clip((data - p2) / (p98 - p2 + 1e-8), 0, 1)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def denormalize(
    data: np.ndarray, original_min: np.ndarray, original_max: np.ndarray
) -> np.ndarray:
    """Reverse min-max normalization."""
    return data * (original_max - original_min) + original_min
