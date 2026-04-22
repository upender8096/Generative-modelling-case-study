from __future__ import annotations

import argparse
from pathlib import Path

import medmnist
import numpy as np
import torch
import torch.nn.functional as F
from medmnist import INFO
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    ensure_dir,
    frechet_distance,
    get_device,
    plot_losses,
    save_image_grid,
    save_json,
    set_seed,
)


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, channels: int = 3, features: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 2, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, channels: int = 3, features: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def load_bloodmnist(batch_size: int = 128):
    data_flag = "bloodmnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    train_dataset = DataClass(split="train", download=True, as_rgb=True)
    test_dataset = DataClass(split="test", download=True, as_rgb=True)

    def collate_normalize(batch):
        imgs = torch.stack([
            torch.tensor(np.array(item[0]), dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
            for item in batch
        ])

        # BloodMNIST is 28x28; resize to 32x32 for this DCGAN architecture
        imgs = F.interpolate(imgs, size=(32, 32), mode="bilinear", align_corners=False)

        labels = torch.tensor(
            [int(np.array(item[1]).squeeze()) for item in batch],
            dtype=torch.long
        )

        return imgs, labels

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_normalize,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_normalize,
    )

    return train_loader, test_loader, info


def pca_features(images: torch.Tensor, n_components: int = 64) -> np.ndarray:
    flat = images.detach().cpu().numpy().reshape(images.shape[0], -1)
    n_components = min(n_components, flat.shape[0], flat.shape[1])
    return PCA(n_components=n_components).fit_transform(flat)


def train(args):
    set_seed(42)
    device = get_device()
    out = ensure_dir(args.output_dir)

    train_loader, test_loader, info = load_bloodmnist(args.batch_size)

    G = Generator(latent_dim=args.latent_dim).to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    g_losses, d_losses = [], []

    for epoch in range(args.epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for real, _ in progress:
            real = real.to(device)
            bs = real.size(0)

            real_targets = torch.ones(bs, 1, device=device)
            fake_targets = torch.zeros(bs, 1, device=device)

            # Train D
            noise = torch.randn(bs, args.latent_dim, 1, 1, device=device)
            fake = G(noise).detach()

            d_real = D(real)
            d_fake = D(fake)
            d_loss = criterion(d_real, real_targets) + criterion(d_fake, fake_targets)

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Train G
            noise = torch.randn(bs, args.latent_dim, 1, 1, device=device)
            fake = G(noise)
            d_fake = D(fake)
            g_loss = criterion(d_fake, real_targets)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            g_losses.append(float(g_loss.item()))
            d_losses.append(float(d_loss.item()))
            progress.set_postfix(
                g_loss=f"{g_loss.item():.4f}",
                d_loss=f"{d_loss.item():.4f}",
            )

        with torch.no_grad():
            samples = G(fixed_noise).cpu()
            save_image_grid(
                samples,
                out / f"samples_epoch_{epoch + 1:03d}.png",
                nrow=8,
                title=f"Generated samples epoch {epoch + 1}",
            )

    plot_losses(g_losses, d_losses, out / "losses.png", title="BloodMNIST DCGAN losses")

    real_batch, _ = next(iter(test_loader))
    real_batch = real_batch[:64]
    save_image_grid(real_batch, out / "real_grid.png", nrow=8, title="Real BloodMNIST samples")

    with torch.no_grad():
        fake_batch = G(torch.randn(64, args.latent_dim, 1, 1, device=device)).cpu()
    save_image_grid(fake_batch, out / "fake_grid.png", nrow=8, title="Generated BloodMNIST samples")

    real_features = pca_features(real_batch)
    fake_features = pca_features(fake_batch)
    approx_fid = frechet_distance(real_features, fake_features)

    metrics = {
        "dataset_description": info["description"],
        "n_classes": len(info["label"]),
        "final_g_loss": g_losses[-1],
        "final_d_loss": d_losses[-1],
        "approx_fid_pca": float(approx_fid),
    }

    save_json(metrics, out / "metrics.json")
    torch.save(G.state_dict(), out / "generator.pt")
    torch.save(D.state_dict(), out / "discriminator.pt")


def main():
    parser = argparse.ArgumentParser(description="Part 2.1 BloodMNIST DCGAN")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="outputs/bloodmnist")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()