import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from .components.gan import Discriminator, Generator


class GANModel(L.LightningModule):
    def __init__(
        self, latent_dim: int = 2, data_dim: int = 2, learning_rate: float = 1e-4
    ):
        """
        GAN model for learning 2D data distribution.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of noise vector
        data_dim : int
            Dimensionality of data points
        learning_rate : float
            Learning rate for optimizers
        """
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim, data_dim)
        self.discriminator = Discriminator(data_dim)

        # Adversarial ground truths
        self.register_buffer("real_label", torch.ones(1))
        self.register_buffer("fake_label", torch.zeros(1))

    def forward(self, z: Tensor) -> Tensor:
        """
        Generate data using generator.

        Parameters
        ----------
        z : Tensor
            Input noise vector

        Returns
        -------
        Tensor
            Generated data points
        """
        return self.generator(z)

    def training_step(
        self, batch: Tensor, batch_idx: int, optimizer_idx: int
    ) -> Tensor:
        """
        Perform training step for GAN.

        Parameters
        ----------
        batch : Tensor
            Real data batch
        batch_idx : int
            Batch index
        optimizer_idx : int
            Optimizer index (0 for generator, 1 for discriminator)

        Returns
        -------
        Tensor
            Computed loss
        """
        real_data = batch
        batch_size = real_data.size(0)

        # Sample noise
        z = torch.randn(batch_size, self.hparams.latent_dim)

        # Generator training
        if optimizer_idx == 0:
            # Generate fake data
            generated_data = self(z)

            # Generator wants to fool discriminator
            g_loss = nn.functional.binary_cross_entropy(
                self.discriminator(generated_data),
                self.real_label.expand_as(self.discriminator(generated_data)),
            )

            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss

        # Discriminator training
        if optimizer_idx == 1:
            # Real data loss
            real_loss = nn.functional.binary_cross_entropy(
                self.discriminator(real_data),
                self.real_label.expand_as(self.discriminator(real_data)),
            )

            # Fake data loss
            generated_data = self(z).detach()
            fake_loss = nn.functional.binary_cross_entropy(
                self.discriminator(generated_data),
                self.fake_label.expand_as(self.discriminator(generated_data)),
            )

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2

            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

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
        z = torch.randn(num_samples, self.hparams.latent_dim)
        return self(z)

    def configure_optimizers(self):
        """
        Configure optimizers for generator and discriminator.

        Returns
        -------
        List
            List of optimizers and their configuration
        """
        g_optimizer = optim.Adam(
            self.generator.parameters(), lr=self.hparams.learning_rate
        )
        d_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=self.hparams.learning_rate
        )

        return [g_optimizer, d_optimizer], []
