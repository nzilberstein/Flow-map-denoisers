"""Linear DCT / IDCT layers as nn.Linear with precomputed orthonormal bases.

Used by the JPEG degradation. Vendored / simplified from zh217/torch-dct.
"""

import numpy as np
import torch
import torch.nn as nn


def _dct_matrix(N: int, norm: str | None = None) -> torch.Tensor:
    """DCT-II basis matrix of shape (N, N). Returns M such that y = M @ x."""
    n = torch.arange(N, dtype=torch.float32)
    k = torch.arange(N, dtype=torch.float32).view(-1, 1)
    M = torch.cos(np.pi / N * (n + 0.5) * k)
    if norm == 'ortho':
        M[0] *= 1.0 / np.sqrt(N)
        M[1:] *= np.sqrt(2.0 / N)
    else:
        M *= 2.0
    return M


class LinearDCT(nn.Linear):
    """1D DCT/IDCT as a Linear layer with frozen, precomputed weights."""

    def __init__(self, in_features: int, type: str, norm: str | None = None, bias: bool = False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super().__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        M = _dct_matrix(self.N, self.norm)
        if self.type == 'dct':
            self.weight.data = M
        elif self.type == 'idct':
            self.weight.data = M.t().contiguous()
        else:
            raise ValueError(f"Unknown LinearDCT type: {self.type}")
        self.weight.requires_grad = False


def apply_linear_2d(x: torch.Tensor, linear_layer: LinearDCT) -> torch.Tensor:
    """Apply a 1D LinearDCT along the last two dims (separable 2D DCT)."""
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)
