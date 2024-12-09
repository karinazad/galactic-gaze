from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from .components.affine_coupling_layer import AffineCouplingLayer


class NormalizingFlow(L.LightningModule):
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
            [AffineCouplingLayer(dim_in=input_dim) for _ in range(num_flows)]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the normalizing flow.
        Returns the latent representation and the log determinant of the Jacobian.

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tuple[Tensor, Tensor]
            Transformed tensor and log determinant of the Jacobian
        """

        z, log_det_total = x, torch.zeros(x.shape[0], device=self.device)

        for flow in self.flows:
            z, log_det_total = flow(z, log_det_total, reverse=False)

        return z, log_det_total

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

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        Perform a single validation step.

        Parameters
        ----------
        batch : Tensor
            Batch of validation data
        batch_idx : int
            Index of the current batch

        Returns
        -------
        Tensor
            Computed negative log-likelihood
        """
        x: Tensor = batch

        z: Tensor
        log_det: Tensor
        z, log_det = self.forward(x)

        log_prob: Tensor = self.base_dist.log_prob(z).sum(dim=-1)
        nll: Tensor = -torch.mean(log_prob + log_det)

        self.log("val_nll", nll, prog_bar=True)
        return nll

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

        z = self.base_dist.sample((num_samples,))
        log_det = torch.zeros(num_samples, device=self.device)

        for flow in reversed(self.flows):
            with torch.no_grad():
                z, log_det = flow(z, log_det, reverse=True)

        return z

    def configure_optimizers(self) -> optim.Adam:
        """
        Configure the optimizer for training.

        Returns
        -------
        optim.Adam
            Configured Adam optimizer
        """
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
