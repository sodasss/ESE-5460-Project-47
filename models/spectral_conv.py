import torch
import torch.nn as nn
import torch.fft as fft


class SpectralConv2d(nn.Module):
    """
    2D spectral convolution using truncated Fourier modes.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (B, in, x, y) * (in, out, x, y) -> (B, out, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        bsz, _, h, w = x.shape

        # Fourier transform
        x_ft = fft.rfft2(x, dim=(-2, -1))

        out_ft = torch.zeros(
            bsz,
            self.out_channels,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, :self.modes2], self.weights2
        )

        x = fft.irfft2(out_ft, s=(h, w), dim=(-2, -1))
        return x
