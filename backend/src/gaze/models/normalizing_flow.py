import torch
import lightning as L
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from typing import Tuple

from .components.normalizing_flow import AffineCouplingLayer


class NormalizingFlowModule(L.LightningModule):
    def __init__(
        self, input_dim: int = 2, num_flows: int = 4, learning_rate: float = 1e-3
    ):
        """
        Initialize the Normalizing Flow Module.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input data
        num_flows : int
            Number of flow layers
        learning_rate : float
            Learning rate for the optimizer
        """
        super().__init__()
        self.save_hyperparameters()

        # Base distribution (standard normal)
        self.base_dist: torch.distributions.Normal = torch.distributions.Normal(
            torch.zeros(input_dim), torch.ones(input_dim)
        )

        # Normalizing Flow
        self.flows: nn.ModuleList = nn.ModuleList(
            [AffineCouplingLayer(input_dim) for _ in range(num_flows)]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the normalizing flow.

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tuple[Tensor, Tensor]
            Transformed tensor and log determinant
        """
        log_det_total: Tensor = torch.zeros(x.size(0), device=x.device)

        for flow in self.flows:
            x, log_det = flow(x)
            log_det_total += log_det

        return x, log_det_total

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        Perform a single training step.

        Parameters
        ----------
        batch : Tensor
            Batch of input data
        batch_idx : int
            Index of the current batch

        Returns
        -------
        Tensor
            Computed loss
        """
        x: Tensor = batch

        z: Tensor
        log_det: Tensor
        z, log_det = self.forward(x)

        log_prob: Tensor = self.base_dist.log_prob(z).sum(dim=-1)
        loss: Tensor = -torch.mean(log_prob + log_det)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _inverse(self, y: Tensor) -> Tensor:
        """
        Inverse transformation through the flow.

        Parameters
        ----------
        y : Tensor
            Input tensor

        Returns
        -------
        Tensor
            Inversely transformed tensor
        """
        for flow in reversed(self.flows):
            y = flow.inverse(y)
        return y

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
        z: Tensor = self.base_dist.sample((num_samples,))
        x: Tensor = self._inverse(z)
        return x

    def configure_optimizers(self) -> optim.Adam:
        """
        Configure the optimizer for training.

        Returns
        -------
        optim.Adam
            Configured Adam optimizer
        """
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
