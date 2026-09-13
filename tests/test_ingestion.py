"""
Tests for the data ingestion module.
"""

import numpy as np
import pytest

from src.ingestion.degradation import create_training_pair, degrade
from src.ingestion.preprocess import normalize
from src.ingestion.tiler import tile_image, untile_image


class TestNormalization:
    """Tests for normalize/denormalize functions."""

    def test_minmax_range(self):
        """Output of minmax normalization should be in [0, 1]."""
        data = np.random.rand(4, 64, 64).astype(np.float32) * 10000
        result = normalize(data, method="minmax")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_percentile_range(self):
        """Output of percentile normalization should be in [0, 1]."""
        data = np.random.rand(4, 64, 64).astype(np.float32) * 10000
        result = normalize(data, method="percentile")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_shape_preserved(self):
        """Normalization should not change array shape."""
        data = np.random.rand(4, 128, 128).astype(np.float32)
        result = normalize(data)
        assert result.shape == data.shape

    def test_invalid_method_raises(self):
        """Invalid normalization method should raise ValueError."""
        data = np.random.rand(4, 64, 64).astype(np.float32)
        with pytest.raises(ValueError):
            normalize(data, method="invalid")


class TestTiler:
    """Tests for tile/untile operations."""

    def test_tile_output_type(self):
        """tile_image should return a list of dicts."""
        image = np.random.rand(4, 256, 256).astype(np.float32)
        patches = tile_image(image, patch_size=128, overlap=0)
        assert isinstance(patches, list)
        assert len(patches) > 0
        assert "patch" in patches[0]

    def test_patch_size(self):
        """Each patch should be (C, patch_size, patch_size)."""
        image = np.random.rand(4, 256, 256).astype(np.float32)
        patches = tile_image(image, patch_size=128, overlap=0)
        for p in patches:
            assert p["patch"].shape == (4, 128, 128)

    def test_tile_untile_roundtrip(self):
        """Tiling then untiling should approximately reconstruct the original."""
        image = np.random.rand(4, 256, 256).astype(np.float32)
        patches = tile_image(image, patch_size=128, overlap=16)
        reconstructed = untile_image(patches, image.shape, patch_size=128)
        # Should be close (not exact due to overlap averaging)
        assert reconstructed.shape == image.shape
        assert np.allclose(reconstructed, image, atol=0.01)


class TestDegradation:
    """Tests for the degradation pipeline."""

    def test_degrade_reduces_resolution(self):
        """Degraded image should be smaller by the scale factor."""
        image = np.random.rand(4, 256, 256).astype(np.float32)
        degraded = degrade(image, scale_factor=3)
        expected_h, expected_w = 256 // 3, 256 // 3
        assert degraded.shape == (4, expected_h, expected_w)

    def test_degrade_output_range(self):
        """Degraded image should stay in [0, 1]."""
        image = np.random.rand(4, 256, 256).astype(np.float32)
        degraded = degrade(image, scale_factor=3, noise_std=0.01)
        assert degraded.min() >= 0.0
        assert degraded.max() <= 1.0

    def test_training_pair_shapes(self):
        """create_training_pair should return matching LR/HR pair."""
        image = np.random.rand(4, 256, 256).astype(np.float32)
        lr, hr = create_training_pair(image, scale_factor=3)
        assert lr.shape[0] == 4  # 4 bands
        assert hr.shape[0] == 4
        assert hr.shape[1] == lr.shape[1] * 3
        assert hr.shape[2] == lr.shape[2] * 3

    def test_no_noise_is_deterministic(self):
        """With noise_std=0, degradation should be deterministic."""
        image = np.random.rand(4, 256, 256).astype(np.float32)
        d1 = degrade(image, scale_factor=3, noise_std=0.0)
        d2 = degrade(image, scale_factor=3, noise_std=0.0)
        assert np.allclose(d1, d2)
