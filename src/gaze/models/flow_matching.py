import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from .components.flow_matching import FlowMatchingNetwork


class FlowMatching(L.LightningModule):
    def __init__(
        self, data_dim: int = 2, learning_rate: float = 1e-3, noise_std: float = 1.0
    ):
        """
        Flow Matching model for learning data distribution.

        Parameters
        ----------
        data_dim : int
            Dimensionality of data points
        learning_rate : float
            Learning rate for optimizer
        noise_std : float
            Standard deviation of initial noise distribution
        """
        super().__init__()
        self.save_hyperparameters()

        self.flow_network = FlowMatchingNetwork(data_dim)
        self.noise_std = noise_std

    def forward(self, x0: Tensor, t: Tensor) -> Tensor:
        """
        Predict velocity field.

        Parameters
        ----------
        x0 : Tensor
            Initial data points
        t : Tensor
            Time condition

        Returns
        -------
        Tensor
            Predicted velocity field
        """
        return self.flow_network(x0, t)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        Perform training step.

        Parameters
        ----------
        batch : Tensor
            Real data batch
        batch_idx : int
            Batch index

        Returns
        -------
        Tensor
            Computed loss
        """
        real_data = batch
        batch_size = real_data.size(0)

        # Sample initial noise distribution
        x0 = torch.randn(batch_size, self.hparams.data_dim) * self.noise_std

        # Random time sampling
        t = torch.rand(batch_size)

        # Compute ODE solution
        x_t = (1 - t).view(-1, 1) * x0 + t.view(-1, 1) * real_data

        # Predict velocity field
        predicted_velocity = self(x_t, t)

        # Compute flow matching loss
        target_velocity = real_data - x0
        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def sample(self, num_samples: int = 100) -> Tensor:
        """
        Generate samples from the model.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate

        Returns
        -------
        Tensor
            Generated samples
        """
        # Sample from initial noise distribution
        x0 = torch.randn(num_samples, self.hparams.data_dim) * self.noise_std

        # Progressive sampling through time
        num_steps = 50
        time_steps = torch.linspace(0, 1, num_steps)

        current_x = x0
        for t in time_steps[1:]:
            velocity = self(current_x, torch.full((current_x.size(0),), t.item()))
            current_x = current_x + velocity * (time_steps[1] - time_steps[0])

        return current_x

    def configure_optimizers(self):
        """
        Configure optimizer for the model.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer
        """
        return optim.Adam(self.flow_network.parameters(), lr=self.hparams.learning_rate)
