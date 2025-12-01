import argparse
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.fno2d import FNO2d
from data.normalize import UnitGaussianNormalizer
from data.dataset import BackwardDataset
from utils.spectral_loss import spectral_reg_loss


def load_data(path: str) -> torch.Tensor:
    """
    Load HDF5 data with dataset /data of shape (N, T, H, W).
    """
    with h5py.File(path, "r") as f:
        data = np.array(f["/data"])
    return torch.from_numpy(data).float()


def main():
    parser = argparse.ArgumentParser(description="Train 2D FNO for backward reconstruction")
    parser.add_argument("--data_path", type=str, default="w_data.h5")
    parser.add_argument("--ntrain", type=int, default=1600)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max_k", type=int, default=5)
    parser.add_argument("--modes1", type=int, default=12)
    parser.add_argument("--modes2", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_path", type=str, default="checkpoints/fno2d.pth")
    parser.add_argument("--normalizer_path", type=str, default="checkpoints/normalizer.pt")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------------------------------------------
    # Load and split data
    # ------------------------------------------------------------------
    data = load_data(args.data_path)  # (N, T, H, W)
    N, T, H, W = data.shape
    print(f"Loaded data: N={N}, T={T}, H={H}, W={W}")

    ntrain = min(args.ntrain, N)
    train_data = data[:ntrain]       # (ntrain, T, H, W)

    # build normalizer using all time steps of training trajectories
    flat_train = train_data.reshape(-1, H, W)
    normalizer = UnitGaussianNormalizer(flat_train)

    train_norm = normalizer.encode(flat_train).reshape(ntrain, T, H, W)

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------
    train_dataset = BackwardDataset(train_norm, max_k=args.max_k)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    # ------------------------------------------------------------------
    # Model & optimizer
    # ------------------------------------------------------------------
    model = FNO2d(args.modes1, args.modes2, args.width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for ep in range(args.epochs):
        model.train()
        acc_loss = 0.0

        for inp, out in train_loader:
            inp = inp.to(device)  # (B, H, W, 3)
            out = out.to(device)  # (B, H, W, 1)

            pred = model(inp)
            loss_data = F.mse_loss(pred, out)
            loss_spec = spectral_reg_loss(pred)
            loss = loss_data + 1e-4 * loss_spec

            opt.zero_grad()
            loss.backward()
            opt.step()

            acc_loss += loss.item()

        avg_loss = acc_loss / len(train_loader)
        print(f"Epoch {ep + 1}/{args.epochs}, loss = {avg_loss:.6f}")

    # ------------------------------------------------------------------
    # Save model & normalizer
    # ------------------------------------------------------------------
    model_dir = os.path.dirname(args.model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    norm_dir = os.path.dirname(args.normalizer_path)
    if norm_dir:
        os.makedirs(norm_dir, exist_ok=True)

    torch.save(model.state_dict(), args.model_path)
    torch.save(normalizer, args.normalizer_path)

    print(f"Saved model to {args.model_path}")
    print(f"Saved normalizer to {args.normalizer_path}")


if __name__ == "__main__":
    main()
