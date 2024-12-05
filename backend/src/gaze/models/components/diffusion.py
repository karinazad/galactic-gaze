import torch
from torch import Tensor, nn


class NoisePredictor(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64):
        """
        Simple MLP to predict noise in diffusion process.

        Parameters
        ----------
        input_dim : int
            Dimensionality of input data
        hidden_dim : int
            Dimensionality of hidden layers
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Predict noise at given timestep.

        Parameters
        ----------
        x : Tensor
            Input data tensor
        t : Tensor
            Timestep tensor

        Returns
        -------
        Tensor
            Predicted noise
        """
        # Combine input with timestep
        input_with_time = torch.cat([x, t], dim=1)
        return self.network(input_with_time)
