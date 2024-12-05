import torch
import lightning as L
from torch.optim import Adam
from torch import Tensor
from typing import List

from .components.vae import Encoder, Decoder
from .components.losses import vae_loss


class VAE(L.LightningModule):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: List[int] = [64, 32],
        latent_dim: int = 2,
        learning_rate: float = 1e-3,
    ):
        """
        Variational Autoencoder Lightning Module for 2D data

        Parameters
        ----------
        input_dim : int
            Dimensionality of input data
        hidden_dims : List[int]
            Hidden layer dimensions
        latent_dim : int
            Dimensionality of latent space
        learning_rate : float
            Optimizer learning rate
        """
        super().__init__()

        self.save_hyperparameters()

        self.encoder: Encoder = Encoder(
            input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim
        )
        self.decoder: Decoder = Decoder(
            latent_dim=latent_dim, hidden_dims=hidden_dims[::-1], output_dim=input_dim
        )

        self.learning_rate: float = learning_rate

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from latent distribution

        Parameters
        ----------
        mu : Tensor
            Mean of latent distribution
        logvar : Tensor
            Log variance of latent distribution

        Returns
        -------
        Tensor
            Sampled latent representation
        """
        std: Tensor = torch.exp(0.5 * logvar)
        eps: Tensor = torch.randn_like(std)
        return mu + eps * std

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        Training step for VAE

        Parameters
        ----------
        batch : Tensor
            Input batch
        batch_idx : int
            Batch index

        Returns
        -------
        Tensor
            Computed loss
        """
        mu: Tensor
        logvar: Tensor
        mu, logvar = self.encoder(batch)
        z: Tensor = self.reparameterize(mu, logvar)
        recon_x: Tensor = self.decoder(z)

        loss: Tensor = vae_loss(recon_x, batch, mu, logvar)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        """
        Validation step for VAE

        Parameters
        ----------
        batch : Tensor
            Input batch
        batch_idx : int
            Batch index
        """
        mu: Tensor
        logvar: Tensor
        mu, logvar = self.encoder(batch)
        z: Tensor = self.reparameterize(mu, logvar)
        recon_x: Tensor = self.decoder(z)

        loss: Tensor = vae_loss(recon_x, batch, mu, logvar)

        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self) -> Adam:
        """
        Configure optimizer for training

        Returns
        -------
        Adam
            Adam optimizer
        """
        return Adam(self.parameters(), lr=self.learning_rate)

    def sample(self, num_samples: int = 1) -> Tensor:
        """
        Sample new points from the learned latent space

        Parameters
        ----------
        num_samples : int
            Number of samples to generate

        Returns
        -------
        Tensor
            Generated data points
        """
        z: Tensor = torch.randn(num_samples, self.encoder.mu_layer.out_features)
        return self.decoder(z)
