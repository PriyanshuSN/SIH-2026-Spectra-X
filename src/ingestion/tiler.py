"""
tiler.py — Split large satellite scenes into patches and reassemble them.

Usage:
    from src.ingestion.tiler import tile_image, untile_image
    patches = tile_image(image, patch_size=256, overlap=16)
    reconstructed = untile_image(patches, original_shape, patch_size=256, overlap=16)
"""

import numpy as np


def tile_image(
    image: np.ndarray,
    patch_size: int = 256,
    overlap: int = 16,
) -> list[dict]:
    """
    Split a (C, H, W) image into overlapping patches for memory-safe inference.

    Args:
        image: Input array of shape (C, H, W).
        patch_size: Size of each square patch.
        overlap: Overlap between adjacent patches (helps avoid edge artifacts).

    Returns:
        List of dicts with keys 'patch' (np.ndarray), 'row', 'col' (position indices).
    """
    C, H, W = image.shape
    step = patch_size - overlap
    patches = []

    for row in range(0, H, step):
        for col in range(0, W, step):
            r_end = min(row + patch_size, H)
            c_end = min(col + patch_size, W)
            r_start = r_end - patch_size
            c_start = c_end - patch_size

            # Clamp to valid range
            r_start = max(0, r_start)
            c_start = max(0, c_start)

            patch = image[:, r_start:r_end, c_start:c_end]

            # Pad if patch is smaller than patch_size (edge case)
            if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                padded = np.zeros((C, patch_size, patch_size), dtype=patch.dtype)
                padded[:, : patch.shape[1], : patch.shape[2]] = patch
                patch = padded

            patches.append({
                "patch": patch,
                "row": r_start,
                "col": c_start,
                "orig_h": r_end - r_start,
                "orig_w": c_end - c_start,
            })

    return patches


def untile_image(
    patches: list[dict],
    original_shape: tuple[int, int, int],
    patch_size: int = 256,
    scale_factor: int = 1,
) -> np.ndarray:
    """
    Reassemble patches back into a full image, handling overlaps by averaging.

    Args:
        patches: List of dicts from tile_image, but with 'patch' replaced by SR output.
        original_shape: (C, H, W) of the original image.
        patch_size: Patch size used during tiling.
        scale_factor: Upscaling factor applied by the SR model.

    Returns:
        Reassembled image of shape (C, H*scale_factor, W*scale_factor).
    """
    C, H, W = original_shape
    out_H, out_W = H * scale_factor, W * scale_factor
    output = np.zeros((C, out_H, out_W), dtype=np.float32)
    counts = np.zeros((1, out_H, out_W), dtype=np.float32)

    # Create a 2D Bartlett (triangular) blending window for seamless stitching
    def get_blending_mask(h, w):
        mask_y = np.bartlett(h)
        mask_x = np.bartlett(w)
        mask_2d = np.outer(mask_y, mask_x)
        # Add a tiny epsilon to prevent zero division
        return mask_2d + 1e-5

    for p in patches:
        r = p["row"] * scale_factor
        c = p["col"] * scale_factor
        ph = p["orig_h"] * scale_factor
        pw = p["orig_w"] * scale_factor
        patch = p["patch"][:, :ph, :pw]

        mask = get_blending_mask(ph, pw)[np.newaxis, ...]

        output[:, r : r + ph, c : c + pw] += patch * mask
        counts[:, r : r + ph, c : c + pw] += mask

    # Normalize by the accumulated blending mask weights
    output /= counts

    return output
