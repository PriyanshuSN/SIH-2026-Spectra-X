"""
preprocess.py — Read GeoTIFF, extract and align B/G/R/NIR bands.

Usage:
    from src.ingestion.preprocess import load_sentinel2_bands
    bands = load_sentinel2_bands("path/to/tile.tif")
"""

import numpy as np


def load_sentinel2_bands(filepath: str, bands: list[int] | None = None) -> np.ndarray:
    """
    Load Sentinel-2 GeoTIFF or standard images and extract bands.
    Automatically pads 3-band RGB images to 4 bands (NIR=0) for compatibility.
    """
    import rasterio
    import numpy as np

    if bands is None:
        bands = [2, 3, 4, 8]  # Blue, Green, Red, NIR for Sentinel-2

    with rasterio.open(filepath) as src:
        # Check if it's a standard RGB image (3 channels)
        if src.count == 3:
            data = src.read([1, 2, 3])
            # Create a dummy NIR band (zeros)
            dummy_nir = np.zeros_like(data[0:1, :, :])
            data = np.concatenate([data, dummy_nir], axis=0)
        else:
            # For 4+ band images, ensure we only read available bands
            valid_bands = [b for b in bands if b <= src.count]
            if not valid_bands:
                valid_bands = list(range(1, src.count + 1))
            data = src.read(valid_bands)
            
            # Pad if less than 4 bands
            if data.shape[0] < 4:
                pad = np.zeros((4 - data.shape[0], data.shape[1], data.shape[2]), dtype=data.dtype)
                data = np.concatenate([data, pad], axis=0)
                
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
