from typing import Callable, List, Tuple
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from .data import generate_galaxy_points


class ToyDataModule(L.LightningDataModule):
    """
    Lightning DataModule for generating 2D point datasets.

    Parameters
    ----------
    point_generator : Callable, optional
        Function to generate points, by default generate_galaxy_points
    batch_size : int, optional
        Size of batches for DataLoader, by default 32
    train_split : float, optional
        Proportion of data used for training, by default 0.8
    noise : float, optional
        Standard deviation of Gaussian noise added to points, by default 0.1
    seed : int, optional
        Random seed for reproducibility, by default 42
    """

    def __init__(
        self,
        data_generating_func: Callable | None = None,
        batch_size: int = 32,
        train_size: float = 0.8,
        noise: float = 0.1,
    ):
        super().__init__()
        self.data_generating_func = data_generating_func
        self.batch_size = batch_size
        self.train_size = train_size
        self.noise = noise

        if self.data_generating_func is None:
            self.data_generating_func = generate_galaxy_points

    def setup(self, stage: str = None):
        data: List[Tuple[float, float]] = self.data_generating_func()

        data: torch.Tensor = torch.tensor(data, dtype=torch.float32)

        data += torch.randn_like(data) * self.noise

        total_points = len(data)
        train_size = int(total_points * self.train_size)

        self.train_data = data[:train_size]
        self.val_data = data[train_size:]

    def train_dataloader(self):
        dataset = SimpleDataset(self.train_data)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        dataset = SimpleDataset(self.val_data)
        return DataLoader(dataset, batch_size=self.batch_size)


class SimpleDataset(Dataset):
    def __init__(self, data: List[Tuple[float, float]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
