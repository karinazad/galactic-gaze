from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class AffineCouplingLayer(nn.Module):
    """Affine Coupling Layer for Normalizing Flows."""

    def __init__(self, dim: int, hidden_dim: int = 64):
        """
        Initialize the Affine Coupling Layer.

        Parameters
        ----------
        dim : int
            Dimension of the input
        hidden_dim : int, optional
            Dimension of the hidden layers, by default 64
        """
        super().__init__()
        self.scale_shift_net: nn.Sequential = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.dim: int = dim

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the Affine Coupling Layer.

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tuple[Tensor, Tensor]
            y : Transformed tensor
            log_det : Log determinant of the transformation
        """
        x1, x2 = torch.chunk(x, 2, dim=-1)
        log_scale, shift = torch.chunk(self.scale_shift_net(x1), 2, dim=-1)

        log_scale = torch.tanh(log_scale)
        y2 = x2 * torch.exp(log_scale) + shift

        y = torch.cat([x1, y2], dim=-1)
        log_det = log_scale.sum(dim=-1)

        return y, log_det

    def inverse(self, y: Tensor) -> Tensor:
        """
        Inverse pass through the Affine Coupling Layer.

        Parameters
        ----------
        y : Tensor
            Input tensor

        Returns
        -------
        Tensor
            x : Inversely transformed tensor
        """
        y1, y2 = torch.chunk(y, 2, dim=-1)
        log_scale, shift = torch.chunk(self.scale_shift_net(y1), 2, dim=-1)

        log_scale = torch.tanh(log_scale)
        x2 = (y2 - shift) * torch.exp(-log_scale)

        x = torch.cat([y1, x2], dim=-1)
        return x
