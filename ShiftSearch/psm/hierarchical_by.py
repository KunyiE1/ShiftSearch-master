from typing import Optional, Sequence

import torch
from torch import nn


class HierarchicalBYScorer(nn.Module):
    """Score PSMs from b/y fragment evidence represented as [g, I].

    Each b/y peak-cleavage cell is encoded as a vector.  A learned scalar
    selector chooses one peak per ion type and cleavage, while the complete
    vector of that peak is retained.  A 1-D CNN then models neighbouring
    cleavage sites before masked mean pooling produces one PSM embedding.
    """

    def __init__(
        self,
        cell_hidden_dim: int = 16,
        cell_embedding_dim: int = 16,
        cleavage_channels: Sequence[int] = (32, 64),
        kernel_size: int = 3,
        head_hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not cleavage_channels:
            raise ValueError("cleavage_channels must not be empty")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so cleavage length is preserved")

        # Bias-free layers and SiLU(0)=0 keep padded cells at zero. Distant
        # valid peaks can carry intensity, but the explicit g factor
        # below still suppresses them before hard peak selection.
        self.cell_encoder = nn.Sequential(
            nn.Conv2d(2, cell_hidden_dim, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(cell_hidden_dim, cell_embedding_dim, kernel_size=1, bias=False),
            nn.SiLU(),
        )
        self.peak_selector = nn.Conv2d(cell_embedding_dim, 1, kernel_size=1, bias=False)

        ion_feature_dim = cell_embedding_dim + 1  # selected vector + selected score
        in_channels = 2 * ion_feature_dim  # b and y remain identifiable after concatenation
        padding = kernel_size // 2
        blocks = []
        for out_channels in cleavage_channels:
            blocks.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        int(out_channels),
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                    ),
                    nn.SiLU(),
                ]
            )
            in_channels = int(out_channels)
        self.cleavage_encoder = nn.ModuleList(blocks)

        self.score_head = nn.Sequential(
            nn.Linear(in_channels, head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def _select_peak(
        self,
        cell_features: torch.Tensor,
        cell_mask: torch.Tensor,
        cleavage_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one complete peak embedding per cleavage.

        Args:
            cell_features: [B, 2, cleavage, peak], channels are [g, I].
            cell_mask: [B, cleavage, peak].
            cleavage_mask: [B, cleavage].

        Returns:
            [B, cell_embedding_dim + 1, cleavage].
        """
        values = self.cell_encoder(cell_features)

        # At initialization the sigmoid term is near 0.5, so g determines the
        # selected peak. Training can refine the intensity/proximity trade-off.
        g = cell_features[:, :1]
        selection_scores = g * torch.sigmoid(self.peak_selector(values))
        selection_scores = selection_scores.squeeze(1)

        lowest = torch.finfo(selection_scores.dtype).min
        masked_scores = selection_scores.masked_fill(~cell_mask, lowest)
        best_score, best_index = masked_scores.max(dim=-1)

        has_peak = cell_mask.any(dim=-1) & cleavage_mask
        best_score = torch.where(has_peak, best_score, torch.zeros_like(best_score))

        gather_index = best_index[:, None, :, None].expand(
            -1,
            values.shape[1],
            -1,
            1,
        )
        selected_values = values.gather(dim=-1, index=gather_index).squeeze(-1)
        selected_values = selected_values * has_peak[:, None].to(selected_values.dtype)

        return torch.cat([selected_values, best_score[:, None]], dim=1)

    def forward(
        self,
        b_cell_features: torch.Tensor,
        y_cell_features: torch.Tensor,
        cell_mask: torch.Tensor,
        cleavage_mask: torch.Tensor,
    ) -> torch.Tensor:
        b_evidence = self._select_peak(b_cell_features, cell_mask, cleavage_mask)
        y_evidence = self._select_peak(y_cell_features, cell_mask, cleavage_mask)
        x = torch.cat([b_evidence, y_evidence], dim=1)

        mask = cleavage_mask[:, None].to(x.dtype)
        x = x * mask
        for layer in self.cleavage_encoder:
            x = layer(x)
            x = x * mask

        pooled = x.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        return self.score_head(pooled).squeeze(-1)


__all__ = ["HierarchicalBYScorer"]
