import torch


def spectral_reg_loss(pred: torch.Tensor, cutoff: float = 0.3) -> torch.Tensor:
    """
    Penalize high-frequency energy in the prediction.

    pred: (B, H, W, 1) or (B, H, W, C)
          This implementation uses the first channel.
    """
    pred2d = pred[..., 0]  # (B, H, W)
    fft2 = torch.fft.fft2(pred2d)
    bsz, h, w = fft2.shape
    cx, cy = int(h * cutoff), int(w * cutoff)

    high = fft2.clone()
    high[:, :cx, :cy] = 0
    high[:, -cx:, :cy] = 0
    high[:, :cx, -cy:] = 0
    high[:, -cx:, -cy:] = 0

    return torch.mean(torch.abs(high) ** 2)
