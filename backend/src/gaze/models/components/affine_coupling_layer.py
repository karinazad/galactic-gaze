import torch
import torch.nn as nn
from typing import Optional

from .flow_masks import create_simple_mask


class SimpleShiftScaleNetwork(nn.Module):
    """Simple feedforward network for learning scale and shift parameters."""

    def __init__(self, input_dim: int, hidden_dim: int = 5):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the scale and shift parameters for the affine coupling layer."""
        return self.model(x)


class AffineCouplingLayer(nn.Module):
    """Affine (scale+shift) coupling layer for learning simple data distributions."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 5, mask: torch.Tensor | None = None
    ):
        super().__init__()

        self.model = SimpleShiftScaleNetwork(input_dim, hidden_dim)
        self.scaling_factor = nn.Parameter(torch.zeros(input_dim))

        if mask is None:
            mask = create_simple_mask(input_dim)

        self.register_buffer("mask", mask)

    def forward(
        self,
        z: torch.Tensor,
        ldj: torch.Tensor,
        reverse: bool = False,
        context: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the coupling layer.

        Parameters:
        -----------
        z : torch.Tensor
            Latent input to the flow
        ldj : torch.Tensor
            The current log-determinant Jacobian of the previous flows.
            The ldj of this layer will be added to this tensor.
        reverse : bool, optional
            If True, we apply the inverse of the layer. Default is False.
        context : torch.Tensor, optional
            Additional context for conditioning the flow

        Returns:
        --------
        tuple[torch.Tensor, torch.Tensor]
            Transformed z and updated ldj
        """
        z_in = z * self.mask

        if context is None:
            nn_out = self.network(z_in)
        else:
            nn_out = self.network(torch.cat([z_in, context], dim=1))

        s, t = nn_out.chunk(2, dim=1)
        s_fac = self.scaling_factor.exp().view(1, -1, 1)
        s = torch.tanh(s / s_fac) * s_fac
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        if not reverse:
            z = (z + t) * torch.exp(s)
            ldj += s.sum(dim=[1])
        else:
            z = (z * torch.exp(-s)) - t
            ldj -= s.sum(dim=[1])

        return z, ldj
