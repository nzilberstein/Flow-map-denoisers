"""Minimal layers shim providing FourierFeatures for k_diffusion.image_transformer_v2."""

import math
import torch
from torch import nn


class FourierFeatures(nn.Module):
    """Random Fourier Features embedding for scalar inputs."""

    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", torch.randn([out_features // 2, in_features]) * std)

    def forward(self, x):
        # x: [..., in_features]
        f = 2 * math.pi * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)
