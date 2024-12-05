import torch
from torch import nn, Tensor


class FlowMatchingNetwork(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64):
        """
        Flow Matching neural network for learning data distribution.

        Parameters
        ----------
        input_dim : int
            Dimensionality of input data
        hidden_dim : int
            Number of hidden units in network layers
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for time condition
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Predict velocity field for flow matching.

        Parameters
        ----------
        x : Tensor
            Input data points
        t : Tensor
            Time condition

        Returns
        -------
        Tensor
            Predicted velocity field
        """
        # Concatenate input with time condition
        xt = torch.cat([x, t.view(-1, 1).expand(-1, x.size(1))], dim=1)
        return self.model(xt)
