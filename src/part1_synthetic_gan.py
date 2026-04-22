
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from common import (
    MLPDiscriminator,
    MLPGenerator,
    ensure_dir,
    plot_losses,
    save_json,
    set_seed,
    get_device,
)


def build_sine_wave(n: int = 3000) -> np.ndarray:
    x = np.random.uniform(-np.pi, np.pi, size=n)
    y = np.sin(x) + 0.1 * np.random.randn(n)
    return np.column_stack([x, y]).astype(np.float32)


def build_spiral(n: int = 3000) -> np.ndarray:
    theta = np.sqrt(np.random.rand(n)) * 4 * np.pi
    r = 0.5 * theta
    x = r * np.cos(theta) + 0.1 * np.random.randn(n)
    y = r * np.sin(theta) + 0.1 * np.random.randn(n)
    data = np.column_stack([x, y]).astype(np.float32)
    return data / np.std(data, axis=0, keepdims=True)


def build_gaussian_mixture(n: int = 3000) -> np.ndarray:
    centers = np.array([[-2, -2], [2, 2], [-2, 2], [2, -2]], dtype=np.float32)
    points = []
    per_cluster = n // len(centers)
    for c in centers:
        pts = c + 0.4 * np.random.randn(per_cluster, 2)
        points.append(pts)
    return np.vstack(points).astype(np.float32)


def build_noisy_curve(n: int = 3000) -> np.ndarray:
    x = np.random.uniform(-3.0, 3.0, size=n)
    eps = np.random.normal(0, 0.15, size=n)
    y = np.sin(2 * x) + 0.3 * np.cos(5 * x) + eps
    return np.column_stack([x, y]).astype(np.float32)


def get_dataset(name: str, n: int = 3000) -> np.ndarray:
    if name == "sine":
        return build_sine_wave(n)
    if name == "spiral":
        return build_spiral(n)
    if name == "gaussian_mixture":
        return build_gaussian_mixture(n)
    if name == "noisy_curve":
        return build_noisy_curve(n)
    raise ValueError(f"Unknown dataset: {name}")


def train_gan(
    data: np.ndarray,
    output_dir: Path,
    latent_dim: int = 8,
    hidden_dim: int = 128,
    depth: int = 2,
    epochs: int = 1000,
    batch_size: int = 128,
    lr: float = 2e-4,
) -> None:
    device = get_device()
    dataset = TensorDataset(torch.tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    G = MLPGenerator(latent_dim=latent_dim, out_dim=2, hidden_dim=hidden_dim, depth=depth).to(device)
    D = MLPDiscriminator(in_dim=2, hidden_dim=hidden_dim, depth=depth).to(device)

    criterion = nn.BCELoss()
    g_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    g_losses, d_losses = [], []

    for epoch in tqdm(range(epochs), desc="Training Part 1 GAN"):
        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            bs = real_batch.size(0)

            real_targets = torch.ones(bs, 1, device=device)
            fake_targets = torch.zeros(bs, 1, device=device)

            # Train discriminator
            z = torch.randn(bs, latent_dim, device=device)
            fake_batch = G(z).detach()

            d_real = D(real_batch)
            d_fake = D(fake_batch)

            d_loss = criterion(d_real, real_targets) + criterion(d_fake, fake_targets)
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Train generator
            z = torch.randn(bs, latent_dim, device=device)
            generated = G(z)
            d_generated = D(generated)
            g_loss = criterion(d_generated, real_targets)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            g_losses.append(float(g_loss.item()))
            d_losses.append(float(d_loss.item()))

    # Save training curves
    plot_losses(g_losses, d_losses, output_dir / "losses.png", title="Part 1 GAN losses")

    # Save comparison plot
    G.eval()
    with torch.no_grad():
        samples = G(torch.randn(len(data), latent_dim, device=device)).cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], s=6, alpha=0.5)
    plt.title("Real data")
    plt.subplot(1, 2, 2)
    plt.scatter(samples[:, 0], samples[:, 1], s=6, alpha=0.5)
    plt.title("Generated data")
    plt.tight_layout()
    plt.savefig(output_dir / "real_vs_fake.png", dpi=200)
    plt.close()

    save_json(
        {
            "epochs": epochs,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "depth": depth,
            "final_g_loss": g_losses[-1],
            "final_d_loss": d_losses[-1],
        },
        output_dir / "metrics.json",
    )

    torch.save(G.state_dict(), output_dir / "generator.pt")
    torch.save(D.state_dict(), output_dir / "discriminator.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 1 synthetic 2D GAN")
    parser.add_argument("--dataset", type=str, default="noisy_curve", choices=["sine", "spiral", "gaussian_mixture", "noisy_curve"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="outputs/part1")
    args = parser.parse_args()

    set_seed(42)
    out = ensure_dir(args.output_dir)
    data = get_dataset(args.dataset)
    train_gan(
        data=data,
        output_dir=out,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
