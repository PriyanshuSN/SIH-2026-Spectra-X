"""
Tests for the SwinIR-Lite model.
"""

import torch

from src.model.swinir import SwinIRLite, build_model


class TestSwinIRLite:
    """Tests for the super-resolution model."""

    def test_forward_pass_shape(self):
        """Output should be (B, 4, H*scale, W*scale)."""
        model = build_model({"scale_factor": 3, "embed_dim": 60, "num_rstb": 2, "depth_per_rstb": 2})
        x = torch.randn(1, 4, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4, 96, 96)

    def test_4band_input_output(self):
        """Model should accept 4 bands and output 4 bands."""
        model = build_model({"num_rstb": 2, "depth_per_rstb": 2})
        x = torch.randn(1, 4, 16, 16)
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == 4  # 4 output bands

    def test_cpu_inference(self):
        """Model must run on CPU (no GPU dependency at demo time)."""
        model = build_model({"num_rstb": 2, "depth_per_rstb": 2}).cpu()
        x = torch.randn(1, 4, 16, 16).cpu()
        with torch.no_grad():
            out = model(x)
        assert out.device.type == "cpu"

    def test_dropout_layers_exist(self):
        """Model should contain Dropout layers for MC-Dropout."""
        model = build_model({"dropout": 0.1})
        has_dropout = any(
            isinstance(m, torch.nn.Dropout) for m in model.modules()
        )
        assert has_dropout, "Model must have Dropout layers for MC-Dropout"

    def test_build_model_defaults(self):
        """build_model with no args should work with sensible defaults."""
        model = build_model()
        assert isinstance(model, SwinIRLite)

    def test_build_model_custom_config(self):
        """build_model should accept custom config overrides."""
        model = build_model({"embed_dim": 32, "num_heads": 2, "num_rstb": 2})
        assert isinstance(model, SwinIRLite)
