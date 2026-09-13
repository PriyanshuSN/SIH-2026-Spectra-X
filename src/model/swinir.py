"""
swinir.py — Lightweight SwinIR-inspired Super-Resolution Model.

Architecture overview:
  1. Shallow Feature Extraction (Conv2d)
  2. Deep Feature Extraction (N × Residual Swin Transformer Blocks)
  3. Reconstruction (PixelShuffle upsampling)

Key differences from full SwinIR (to fit free-tier GPU):
  - Fewer RSTB blocks (4-6 instead of 6-20)
  - Smaller embedding dimension (60-96 instead of 180)
  - Fewer attention heads (4-6 instead of 6-8)
  - Dropout layers included for MC-Dropout uncertainty estimation

Input: (B, 4, H, W) — 4 bands (B/G/R/NIR)
Output: (B, 4, H*scale, W*scale) — upscaled 4 bands
"""

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Feed-forward network used in each transformer block."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA)."""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """A single Swin Transformer block with optional window shifting."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift: bool = False,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift = shift
        self.shift_size = window_size // 2 if shift else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x, H, W):
        B, _L, C = x.shape
        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Partition windows
        x = _window_partition(x, self.window_size)  # (num_windows*B, ws*ws, C)

        # W-MSA
        x = self.attn(x)

        # Merge windows
        x = _window_reverse(x, self.window_size, H, W)  # (B, H, W, C)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block — a stack of Swin blocks + residual conv."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int = 6,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift=(i % 2 == 1),  # Alternate between W-MSA and SW-MSA
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, H, W):
        residual = x
        for blk in self.blocks:
            x = blk(x, H, W)
        # Reshape for conv
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).view(B, L, C)
        return x + residual


class SwinIRLite(nn.Module):
    """
    Lightweight SwinIR for satellite super-resolution.

    Designed to:
    - Process 4 bands jointly (B/G/R/NIR)
    - Include dropout for MC-Dropout uncertainty estimation
    - Fit in free-tier GPU memory (Colab T4 / Kaggle P100)
    - Run on CPU as fallback (slower but works)

    Args:
        in_channels: Number of input bands (default: 4 for B/G/R/NIR).
        embed_dim: Embedding dimension (default: 60, keep small for free GPU).
        num_heads: Number of attention heads (default: 4).
        num_rstb: Number of RSTB blocks (default: 4, keep small).
        depth_per_rstb: Transformer blocks per RSTB (default: 4).
        window_size: Attention window size (default: 8).
        scale_factor: Upscaling factor (default: 3, i.e. 10m → ~3.3m).
        dropout: Dropout rate for MC-Dropout (default: 0.1).
    """

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 60,
        num_heads: int = 4,
        num_rstb: int = 4,
        depth_per_rstb: int = 4,
        window_size: int = 8,
        scale_factor: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.window_size = window_size

        # 1. Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction (RSTB blocks)
        self.rstb_blocks = nn.ModuleList([
            RSTB(
                dim=embed_dim,
                num_heads=num_heads,
                depth=depth_per_rstb,
                window_size=window_size,
                dropout=dropout,
            )
            for _ in range(num_rstb)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # 3. Reconstruction (PixelShuffle upsampling)
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, (scale_factor ** 2) * in_channels, 3, 1, 1),
            nn.PixelShuffle(scale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 4, H, W).

        Returns:
            Super-resolved output of shape (B, 4, H*scale, W*scale).
        """
        B, _C, H, W = x.shape

        # Pad to be divisible by window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        _, _, Hp, Wp = x.shape

        # Shallow features
        shallow = self.conv_first(x)

        # Deep features
        feat = shallow.permute(0, 2, 3, 1).view(B, Hp * Wp, -1)  # (B, H*W, C)
        for rstb in self.rstb_blocks:
            feat = rstb(feat, Hp, Wp)
        feat = self.norm(feat)
        feat = feat.view(B, Hp, Wp, -1).permute(0, 3, 1, 2)  # (B, C, H, W)

        # Residual connection
        feat = self.conv_after_body(feat) + shallow

        # Global bicubic base residual
        bicubic_base = F.interpolate(
            x[:, :, :H, :W],
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=False,
        )

        # Upsample residual
        out = self.upsample(feat)

        # Remove padding
        out = out[:, :, : H * self.scale_factor, : W * self.scale_factor]

        # Combine learned residual with bicubic base
        return out + bicubic_base


# ─── Helper functions ───────────────────────────────────────────────────────


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition (B, H, W, C) into windows of (num_windows*B, ws*ws, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def _window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition back to (B, H, W, C)."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def build_model(config: dict | None = None) -> SwinIRLite:
    """
    Build model with optional config override.

    Default config is tuned for free-tier GPU (Colab T4 / Kaggle P100).
    """
    defaults = {
        "in_channels": 4,
        "embed_dim": 60,
        "num_heads": 4,
        "num_rstb": 7,  # Upgraded to 1.54M parameters!
        "depth_per_rstb": 4,
        "window_size": 8,
        "scale_factor": 3,
        "dropout": 0.1,
    }
    if config:
        defaults.update(config)
    return SwinIRLite(**defaults)
