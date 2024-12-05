from torch import nn, Tensor


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 2, output_dim: int = 2):
        """
        Generator network for 2D data generation.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of input noise vector
        output_dim : int
            Dimensionality of generated data
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Generate data from noise vector.

        Parameters
        ----------
        z : Tensor
            Input noise vector

        Returns
        -------
        Tensor
            Generated data points
        """
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 2):
        """
        Discriminator network for distinguishing real/fake data.

        Parameters
        ----------
        input_dim : int
            Dimensionality of input data
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Classify input as real or fake.

        Parameters
        ----------
        x : Tensor
            Input data points

        Returns
        -------
        Tensor
            Probability of input being real
        """
        return self.model(x)
