import torch
from torch import Tensor


def vae_loss(
    recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor, reduction: str = "mean"
) -> Tensor:
    """
    Compute VAE loss: Reconstruction Loss + KL Divergence

    Parameters
    ----------
    recon_x : Tensor
        Reconstructed input
    x : Tensor
        Original input
    mu : Tensor
        Mean of the latent distribution
    logvar : Tensor
        Log variance of the latent distribution
    reduction : str, optional
        Reduction method for loss ('mean' or 'sum'), by default 'mean'

    Returns
    -------
    Tensor
        Total VAE loss
    """
    # Reconstruction loss (MSE)
    recon_loss: Tensor = torch.nn.functional.mse_loss(recon_x, x, reduction=reduction)

    # KL Divergence loss
    # KL(q(z|x) || N(0,1))
    kl_loss: Tensor = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if reduction == "mean":
        kl_loss = kl_loss.mean()

    return recon_loss + kl_loss
