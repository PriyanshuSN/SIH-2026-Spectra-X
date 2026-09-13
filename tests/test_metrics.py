"""
Tests for the metrics module.
"""

import numpy as np

from src.metrics.compute import (
    compute_all_metrics,
    compute_ergas,
    compute_psnr,
    compute_sam,
    compute_ssim,
)


class TestPSNR:
    def test_identical_images(self):
        """PSNR of identical images should be very high (inf or near-inf)."""
        img = np.random.rand(4, 64, 64).astype(np.float32)
        psnr = compute_psnr(img, img)
        assert psnr > 50  # Effectively infinite for identical images

    def test_different_images(self):
        """PSNR of different images should be finite and positive."""
        ref = np.random.rand(4, 64, 64).astype(np.float32)
        tgt = np.random.rand(4, 64, 64).astype(np.float32)
        psnr = compute_psnr(ref, tgt)
        assert 0 < psnr < 100


class TestSSIM:
    def test_identical_images(self):
        """SSIM of identical images should be 1.0."""
        img = np.random.rand(4, 64, 64).astype(np.float32)
        ssim = compute_ssim(img, img)
        assert abs(ssim - 1.0) < 1e-5

    def test_range(self):
        """SSIM should be between 0 and 1 for reasonable inputs."""
        ref = np.random.rand(4, 64, 64).astype(np.float32)
        tgt = np.random.rand(4, 64, 64).astype(np.float32)
        ssim = compute_ssim(ref, tgt)
        assert -1.0 <= ssim <= 1.0


class TestSAM:
    def test_identical_images(self):
        """SAM of identical images should be 0 degrees."""
        img = np.random.rand(4, 64, 64).astype(np.float32) + 0.1  # avoid zero
        sam = compute_sam(img, img)
        assert sam < 0.01  # Essentially zero

    def test_positive_output(self):
        """SAM should always be non-negative."""
        ref = np.random.rand(4, 64, 64).astype(np.float32) + 0.1
        tgt = np.random.rand(4, 64, 64).astype(np.float32) + 0.1
        sam = compute_sam(ref, tgt)
        assert sam >= 0


class TestERGAS:
    def test_identical_images(self):
        """ERGAS of identical images should be 0."""
        img = np.random.rand(4, 64, 64).astype(np.float32) + 0.1
        ergas = compute_ergas(img, img)
        assert ergas < 0.01

    def test_positive_output(self):
        """ERGAS should be non-negative."""
        ref = np.random.rand(4, 64, 64).astype(np.float32) + 0.1
        tgt = np.random.rand(4, 64, 64).astype(np.float32) + 0.1
        ergas = compute_ergas(ref, tgt)
        assert ergas >= 0


class TestComputeAll:
    def test_returns_all_keys(self):
        """compute_all_metrics should return all four metrics."""
        ref = np.random.rand(4, 64, 64).astype(np.float32)
        tgt = np.random.rand(4, 64, 64).astype(np.float32)
        metrics = compute_all_metrics(ref, tgt)
        assert "psnr" in metrics
        assert "ssim" in metrics
        assert "sam" in metrics
        assert "ergas" in metrics
