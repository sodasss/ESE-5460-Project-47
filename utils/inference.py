import numpy as np
import torch

from data.normalize import UnitGaussianNormalizer  # for type hints only


def evaluate_backward(
    model: torch.nn.Module,
    traj: np.ndarray,
    normalizer: UnitGaussianNormalizer,
    device: torch.device | None = None,
    sigma: float = 0.05,
    max_k: int = 5,
) -> np.ndarray:
    """
    Backward reconstruction:
    given full trajectory w(t) and noisy final state, reconstruct initial field w0.

    traj: (T, H, W) in physical space (same scale as data in HDF5)
    """
    if device is None:
        device = next(model.parameters()).device

    T, H, W = traj.shape

    # noisy final state in physical space
    wT_phys = traj[-1] + sigma * np.random.randn(H, W)

    # normalize final state
    wT_norm = normalizer.encode(torch.from_numpy(wT_phys).float()).cpu().numpy()

    xs = np.linspace(0, 1, H)
    ys = np.linspace(0, 1, W)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    grid = np.stack([X, Y], axis=-1)  # (H, W, 2)

    w_est = wT_norm
    t = T - 1

    model.eval()
    with torch.no_grad():
        while t > 0:
            k = min(max_k, t)
            inp = np.zeros((H, W, 3), dtype=np.float32)
            inp[..., 0] = w_est
            inp[..., 1:] = grid

            x = torch.from_numpy(inp).unsqueeze(0).to(device)  # (1, H, W, 3)
            pred_norm = model(x)[0].cpu().numpy().squeeze()    # (H, W)

            w_est = pred_norm
            t -= k

    w_est_flat = torch.from_numpy(w_est).view(1, H, W)
    w0_pred = normalizer.decode(w_est_flat).cpu().numpy()[0]  # (H, W)

    return w0_pred
