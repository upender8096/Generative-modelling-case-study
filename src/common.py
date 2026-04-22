
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import linalg
from torch import nn


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict, path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_losses(g_losses, d_losses, path: str | Path, title: str = "Training losses") -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(g_losses, label="Generator loss")
    plt.plot(d_losses, label="Discriminator loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_image_grid(images: torch.Tensor, nrow: int = 8, normalize: bool = True) -> np.ndarray:
    """
    Create a simple grid without relying on torchvision save helpers.
    images: [N, C, H, W]
    """
    images = images.detach().cpu()
    if normalize:
        min_v = images.min()
        max_v = images.max()
        images = (images - min_v) / (max_v - min_v + 1e-8)

    n, c, h, w = images.shape
    rows = math.ceil(n / nrow)
    grid = torch.zeros((c, rows * h, nrow * w))
    for idx in range(n):
        r = idx // nrow
        col = idx % nrow
        grid[:, r * h:(r + 1) * h, col * w:(col + 1) * w] = images[idx]

    grid = grid.permute(1, 2, 0).numpy()
    if grid.shape[-1] == 1:
        grid = grid[..., 0]
    return grid


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 8, title: Optional[str] = None) -> None:
    grid = make_image_grid(images, nrow=nrow, normalize=True)
    plt.figure(figsize=(8, 8))
    if grid.ndim == 2:
        plt.imshow(grid, cmap="gray")
    else:
        plt.imshow(grid)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


class MLPGenerator(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int, hidden_dim: int = 128, depth: int = 2):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MLPDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, depth: int = 2):
        super().__init__()
        layers = []
        cur = in_dim
        for _ in range(depth):
            layers += [nn.Linear(cur, hidden_dim), nn.LeakyReLU(0.2, inplace=True)]
            cur = hidden_dim
        layers += [nn.Linear(cur, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def frechet_distance(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """
    Generic FID-style distance for two feature matrices.
    This is exact for the chosen feature space, but its meaning depends on the feature extractor used.
    """
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


@dataclass
class TrainSummary:
    final_g_loss: float
    final_d_loss: float
    extra_metrics: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "final_g_loss": self.final_g_loss,
            "final_d_loss": self.final_d_loss,
            "extra_metrics": self.extra_metrics,
        }
