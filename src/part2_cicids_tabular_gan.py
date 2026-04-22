from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from common import (
    MLPDiscriminator,
    MLPGenerator,
    ensure_dir,
    get_device,
    plot_losses,
    save_json,
    set_seed,
)


def load_data(csv_path: str, max_rows: int = 100000):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError("Label column not found in CSV.")

    labels = df["Label"].astype(str).str.upper()

    # Keep only BENIGN and DoS/DDoS rows
    mask = labels.str.contains("BENIGN") | labels.str.contains("DOS") | labels.str.contains("DDOS")
    df = df.loc[mask].copy()
    labels = labels.loc[df.index]

    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
        labels = labels.loc[df.index]

    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = numeric_df.to_numpy(dtype=np.float32)
    X = StandardScaler().fit_transform(X).astype(np.float32)

    return X, labels.to_numpy(), numeric_df.columns.tolist()


def plot_pca(real_data: np.ndarray, fake_data: np.ndarray, save_path: Path):
    pca = PCA(n_components=2, random_state=42)
    combined = np.vstack([real_data, fake_data])
    reduced = pca.fit_transform(combined)

    real_proj = reduced[: len(real_data)]
    fake_proj = reduced[len(real_data):]

    plt.figure(figsize=(7, 6))
    plt.scatter(real_proj[:, 0], real_proj[:, 1], s=8, alpha=0.5, label="Real")
    plt.scatter(fake_proj[:, 0], fake_proj[:, 1], s=8, alpha=0.5, label="Fake")
    plt.title("Real vs Fake Traffic Samples (PCA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def train(args):
    set_seed(42)
    device = get_device()
    output_dir = ensure_dir(args.output_dir)

    X, labels, feature_names = load_data(args.input_csv, args.max_rows)

    data_tensor = torch.tensor(X)
    loader = DataLoader(
        TensorDataset(data_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    input_dim = X.shape[1]

    G = MLPGenerator(
        latent_dim=args.latent_dim,
        out_dim=input_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth
    ).to(device)

    D = MLPDiscriminator(
        in_dim=input_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth
    ).to(device)

    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    g_losses = []
    d_losses = []

    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for (real_batch,) in progress:
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_batch = G(noise).detach()

            d_real = D(real_batch)
            d_fake = D(fake_batch)

            d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            noise = torch.randn(batch_size, args.latent_dim, device=device)
            generated_batch = G(noise)

            g_loss = criterion(D(generated_batch), real_labels)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            progress.set_postfix(
                g_loss=f"{g_loss.item():.4f}",
                d_loss=f"{d_loss.item():.4f}"
            )

    plot_losses(g_losses, d_losses, output_dir / "losses.png", title="CICIDS GAN Losses")

    with torch.no_grad():
        sample_size = min(2000, len(X))
        real_sample = X[:sample_size]
        noise = torch.randn(sample_size, args.latent_dim, device=device)
        fake_sample = G(noise).cpu().numpy()

    plot_pca(real_sample, fake_sample, output_dir / "pca_real_vs_fake.png")

    real_stats = pd.DataFrame(real_sample, columns=feature_names).describe().T[["mean", "std"]]
    fake_stats = pd.DataFrame(fake_sample, columns=feature_names).describe().T[["mean", "std"]]

    comparison = real_stats.join(fake_stats, lsuffix="_real", rsuffix="_fake")
    comparison["abs_mean_diff"] = (comparison["mean_real"] - comparison["mean_fake"]).abs()
    comparison.sort_values("abs_mean_diff", ascending=False).head(25).to_csv(
        output_dir / "feature_stat_comparison_top25.csv"
    )

    metrics = {
        "n_rows_used": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "final_g_loss": float(g_losses[-1]),
        "final_d_loss": float(d_losses[-1]),
        "class_counts": pd.Series(labels).value_counts().to_dict(),
    }

    save_json(metrics, output_dir / "metrics.json")
    torch.save(G.state_dict(), output_dir / "generator.pt")
    torch.save(D.state_dict(), output_dir / "discriminator.pt")


def main():
    parser = argparse.ArgumentParser(description="Simplified CICIDS Tabular GAN")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_rows", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default="outputs/cicids")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()