from typing import Iterable, Optional, Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def masked_global_average_pool2d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=x.dtype)
    masked_x = x * mask
    denom = mask.sum(dim=(-2, -1)).clamp(min=1.0)
    return masked_x.sum(dim=(-2, -1)) / denom


class PSM2DCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128),
        kernel_size: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        ppm_hidden_dim: int = 16,
        ppm_output_dim: int = 16,
    ) -> None:
        super().__init__()
        if not isinstance(channels, Iterable) or len(tuple(channels)) == 0:
            raise ValueError("channels must be a non-empty sequence")

        channel_list = list(channels)
        blocks = []
        prev_channels = in_channels
        for out_channels in channel_list:
            blocks.append(ConvBlock(prev_channels, out_channels, kernel_size=kernel_size))
            prev_channels = out_channels

        self.features = nn.ModuleList(blocks)
        self.ppm_mlp = nn.Sequential(
            nn.Linear(1, ppm_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ppm_hidden_dim, ppm_output_dim),
        )
        self.trunk = nn.Sequential(
            nn.Linear(prev_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Linear(hidden_dim + ppm_output_dim, 1)

    def forward(self, pair_tensor: torch.Tensor, pair_mask: Optional[torch.Tensor] = None, ppm_error: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pair_mask is None:
            pair_mask = torch.ones(
                pair_tensor.shape[0],
                pair_tensor.shape[-2],
                pair_tensor.shape[-1],
                device=pair_tensor.device,
                dtype=pair_tensor.dtype,
            )

        if pair_mask.dim() == 3:
            mask_channel = pair_mask.unsqueeze(1).to(dtype=pair_tensor.dtype)
        else:
            mask_channel = pair_mask.to(dtype=pair_tensor.dtype)

        x = pair_tensor.clone()
        x[:, :2] = x[:, :2] * mask_channel

        for block in self.features:
            x = block(x)

        x = masked_global_average_pool2d(x, pair_mask)
        x = self.trunk(x)
        if ppm_error is None:
            ppm_error = torch.zeros(pair_tensor.shape[0], 1, device=pair_tensor.device, dtype=pair_tensor.dtype)
        ppm_feat = self.ppm_mlp(ppm_error.view(-1, 1))
        x = torch.cat([x, ppm_feat], dim=1)
        logits = self.score_head(x).squeeze(-1)
        return logits
