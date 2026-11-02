from typing import Dict

import torch
from torch import nn


class PSMBinaryLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        labels = labels.float().view_as(logits)
        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        true_positive = (preds * labels).sum()
        accuracy = (preds == labels).float().mean()
        precision = true_positive / (preds.sum() + 1e-8)
        recall = true_positive / (labels.sum() + 1e-8)
        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }
