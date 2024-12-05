import torch.nn as nn
from torch import Tensor
from typing import Sequence, Tuple, List


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = [64, 32],
        latent_dim: int = 2,
    ):
        """
        2D Encoder for VAE

        Parameters
        ----------
        input_dim : int, optional
            Dimensionality of input data, by default 2
        hidden_dims : List[int], optional
            List of hidden layer dimensions, by default [64, 32]
        latent_dim : int, optional
            Dimensionality of latent space, by default 2
        """
        super().__init__()

        # Construct hidden layers
        layers: List[nn.Module] = []
        prev_dim: int = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim

        self.hidden_layers: nn.Sequential = nn.Sequential(*layers)

        # Mu and log variance layers
        self.mu_layer: nn.Linear = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer: nn.Linear = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of encoder

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tuple[Tensor, Tensor]
            mu : Mean of latent distribution
            logvar : Log variance of latent distribution
        """
        h: Tensor = self.hidden_layers(x)
        mu: Tensor = self.mu_layer(h)
        logvar: Tensor = self.logvar_layer(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dims: List[int] = [32, 64],
        output_dim: int = 2,
    ):
        """
        2D Decoder for VAE

        Parameters
        ----------
        latent_dim : int, optional
            Dimensionality of latent space, by default 2
        hidden_dims : List[int], optional
            List of hidden layer dimensions (reversed from encoder), by default [32, 64]
        output_dim : int, optional
            Dimensionality of output data, by default 2
        """
        super().__init__()

        # Construct hidden layers
        layers: List[nn.Module] = []
        prev_dim: int = latent_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim

        # Final reconstruction layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.decoder_layers: nn.Sequential = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """
        Forward pass of decoder

        Parameters
        ----------
        z : Tensor
            Latent space representation

        Returns
        -------
        Tensor
            Reconstructed data point
        """
        return self.decoder_layers(z)
