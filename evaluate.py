import argparse

import h5py
import numpy as np
import torch

from models.fno2d import FNO2d
from utils.inference import evaluate_backward
from utils.visualization import visualize


def load_data(path: str) -> np.ndarray:
    """
    Load HDF5 data with dataset /data of shape (N, T, H, W).
    """
    with h5py.File(path, "r") as f:
        data = np.array(f["/data"])
    return data.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained FNO model")
    parser.add_argument("--data_path", type=str, default="w_data.h5")
    parser.add_argument("--model_path", type=str, default="checkpoints/fno2d.pth")
    parser.add_argument("--normalizer_path", type=str, default="checkpoints/normalizer.pt")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--modes1", type=int, default=12)
    parser.add_argument("--modes2", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--max_k", type=int, default=5)
    parser.add_argument("--no_plot", action="store_true", help="Disable matplotlib plotting")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data = load_data(args.data_path)  # (N, T, H, W)
    N, T, H, W = data.shape
    print(f"Loaded data: N={N}, T={T}, H={H}, W={W}")

    assert 0 <= args.sample_index < N, "sample_index out of range"
    traj = data[args.sample_index]  # (T, H, W)
    real_w0 = traj[0]

    # ------------------------------------------------------------------
    # Load model & normalizer
    # ------------------------------------------------------------------
    model = FNO2d(args.modes1, args.modes2, args.width).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded model from {args.model_path}")

    normalizer = torch.load(args.normalizer_path, map_location="cpu")
    print(f"Loaded normalizer from {args.normalizer_path}")

    # ------------------------------------------------------------------
    # Backward reconstruction
    # ------------------------------------------------------------------
    pred_w0 = evaluate_backward(
        model=model,
        traj=traj,
        normalizer=normalizer,
        device=device,
        sigma=args.sigma,
        max_k=args.max_k,
    )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    if not args.no_plot:
        visualize(real_w0, pred_w0)


if __name__ == "__main__":
    main()
