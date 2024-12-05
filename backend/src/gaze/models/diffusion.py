import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch import Tensor

from .components.diffusion import NoisePredictor


class DiffusionModel(L.LightningModule):
    def __init__(
        self,
        input_dim: int = 2,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        learning_rate: float = 1e-3,
    ):
        """
        Initialize Diffusion Model for 2D distribution learning.

        Parameters
        ----------
        input_dim : int
            Dimensionality of input data
        timesteps : int
            Number of diffusion timesteps
        beta_start : float
            Starting noise schedule value
        beta_end : float
            Ending noise schedule value
        learning_rate : float
            Optimizer learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        # Noise prediction network
        self.noise_predictor = NoisePredictor(input_dim)

        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through noise predictor.

        Parameters
        ----------
        x : Tensor
            Input data tensor

        Returns
        -------
        Tensor
            Predicted noise
        """
        batch_size = x.shape[0]

        # Random timestep for each sample in batch
        t = (
            torch.randint(
                0, self.hparams.timesteps, (batch_size, 1), device=x.device
            ).float()
            / self.hparams.timesteps
        )

        # Add noise to input
        noise = torch.randn_like(x)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[
            int(t.item() * self.hparams.timesteps)
        ].view(1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[
            int(t.item() * self.hparams.timesteps)
        ].view(1, 1)

        noisy_x = sqrt_alpha_cumprod_t * x + sqrt_one_minus_alpha_cumprod_t * noise

        # Predict noise
        predicted_noise = self.noise_predictor(noisy_x, t)

        return predicted_noise, noise

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
        predicted_noise, true_noise = self(x)

        # MSE loss between predicted and true noise
        loss: Tensor = nn.functional.mse_loss(predicted_noise, true_noise)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def sample(self, num_samples: int = 100) -> Tensor:
        """
        Generate samples.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate

        Returns
        -------
        Tensor
            Generated samples
        """
        # Start with pure noise
        x = torch.randn(num_samples, self.hparams.input_dim)

        # Reverse diffusion process
        for t in reversed(range(self.hparams.timesteps)):
            z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            t_tensor = torch.full((num_samples, 1), t / self.hparams.timesteps)

            # Predict noise
            predicted_noise = self.noise_predictor(x, t_tensor)

            # Denoise step
            x = (
                x - predicted_noise * self.sqrt_one_minus_alphas_cumprod[t]
            ) / torch.sqrt(self.alphas[t]) + z * torch.sqrt(self.betas[t])

        return x

    def configure_optimizers(self):
        """
        Configure optimizer for training.

        Returns
        -------
        torch.optim.Adam
            Configured Adam optimizer
        """
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
