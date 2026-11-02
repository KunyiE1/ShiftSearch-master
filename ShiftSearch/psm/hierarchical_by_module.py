import math
from typing import Dict

import lightning.pytorch as pl
import torch
from torch import optim

from ShiftSearch.constants import H2O, PROTON
from ShiftSearch.psm.hierarchical_by import HierarchicalBYScorer
from ShiftSearch.psm.loss import PSMBinaryLoss


def build_warmup_cosine_scheduler(
    optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
):
    if total_steps <= 0:
        return None
    warmup_steps = max(0, min(warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps == warmup_steps:
            return min_lr_ratio
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class PSMHierarchicalBY(pl.LightningModule):
    """Lightning module for the first b/y hierarchical scoring architecture."""

    def __init__(self, model_params: Dict, train_params: Dict, meta_params: Dict) -> None:
        super().__init__()
        self.train_params = dict(train_params)
        self.meta_params = dict(meta_params)
        self.lr = float(train_params.get("lr", 1e-3))
        self.n_epochs = int(train_params.get("n_epochs", 1))
        self.warmup_steps_cfg = train_params.get("warmup_steps")
        self.warmup_epochs = train_params.get("warmup_epochs")
        self.min_lr_ratio = float(train_params.get("min_lr_ratio", 0.0))
        self.log_interval = int(train_params.get("log_interval", 0))

        self.max_mz = float(meta_params["max_mz"])
        self.max_pep_len = int(meta_params["max_pep_len"])
        self.max_n_peaks = int(meta_params["max_npeaks"])
        self.fragment_tolerance_ppm = float(meta_params.get("fragment_tolerance_ppm", 20.0))
        if self.fragment_tolerance_ppm <= 0:
            raise ValueError("fragment_tolerance_ppm must be positive")

        self.encoder = HierarchicalBYScorer(**dict(model_params))
        self.loss_fn = PSMBinaryLoss()
        self.save_hyperparameters(
            {
                "model_params": dict(model_params),
                "train_params": self.train_params,
                "meta_params": self.meta_params,
            }
        )

    def prepare_spectrum_batch(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        raw_peak_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select and normalize peaks once for each distinct spectrum.

        Search reuses the returned tensors for every candidate of a spectrum,
        instead of repeating top-k selection and normalization per PSM chunk.
        """
        device = next(self.parameters()).device
        mz = mz.to(device=device, dtype=torch.float32)
        intensity = intensity.to(device=device, dtype=torch.float32)
        raw_peak_mask = raw_peak_mask.to(device=device, dtype=torch.bool)
        valid = raw_peak_mask & (mz > 0) & (mz < self.max_mz) & (intensity > 0)
        masked_intensity = intensity.masked_fill(~valid, float("-inf"))
        k = min(self.max_n_peaks, masked_intensity.shape[1])
        top_index = torch.topk(masked_intensity, k=k, dim=1).indices
        top_mz = mz.gather(1, top_index)
        top_intensity = intensity.gather(1, top_index)
        top_valid = valid.gather(1, top_index)

        if k < self.max_n_peaks:
            pad = self.max_n_peaks - k
            top_mz = torch.cat([top_mz, top_mz.new_zeros(top_mz.shape[0], pad)], dim=1)
            top_intensity = torch.cat(
                [top_intensity, top_intensity.new_zeros(top_intensity.shape[0], pad)],
                dim=1,
            )
            top_valid = torch.cat(
                [top_valid, torch.zeros(top_valid.shape[0], pad, device=device, dtype=torch.bool)],
                dim=1,
            )

        # Preserve the normalization used by the existing architecture.
        normalized_intensity = torch.sqrt(top_intensity.clamp_min(0.0))
        normalized_intensity = normalized_intensity.masked_fill(~top_valid, 0.0)
        normalized_intensity = normalized_intensity / normalized_intensity.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-8)

        return top_mz, normalized_intensity, top_valid

    def build_candidate_features(
        self,
        top_mz: torch.Tensor,
        normalized_intensity: torch.Tensor,
        top_valid: torch.Tensor,
        prefix_masses: torch.Tensor,
        peptide_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build b/y evidence for a batch of spectrum-candidate pairs."""
        device = next(self.parameters()).device
        top_mz = top_mz.to(device=device, dtype=torch.float32)
        normalized_intensity = normalized_intensity.to(device=device, dtype=torch.float32)
        top_valid = top_valid.to(device=device, dtype=torch.bool)
        prefix_masses = prefix_masses.to(device=device, dtype=torch.float32)
        peptide_length = peptide_length.to(device=device, dtype=torch.long)

        batch_size = prefix_masses.shape[0]
        for name, tensor in (
            ("top_mz", top_mz),
            ("normalized_intensity", normalized_intensity),
            ("top_valid", top_valid),
        ):
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"{name} batch size {tensor.shape[0]} does not match "
                    f"candidate batch size {batch_size}"
                )

        n_cleavages = self.max_pep_len - 1
        prefix = prefix_masses[:, 1 : 1 + n_cleavages]
        total_residue_mass = prefix_masses.gather(1, peptide_length[:, None])

        # Row i represents one peptide bond: b_i and its complementary y_(L-i).
        theoretical_b_mz = prefix + PROTON
        theoretical_y_mz = total_residue_mass - prefix + H2O + PROTON

        b_delta = top_mz[:, None, :] - theoretical_b_mz[:, :, None]
        y_delta = top_mz[:, None, :] - theoretical_y_mz[:, :, None]
        b_tolerance = theoretical_b_mz[:, :, None].clamp_min(1e-6) * (
            self.fragment_tolerance_ppm * 1e-6
        )
        y_tolerance = theoretical_y_mz[:, :, None].clamp_min(1e-6) * (
            self.fragment_tolerance_ppm * 1e-6
        )

        # Values beyond eight tolerance units have negligible Gaussian support.
        b_r = (b_delta / b_tolerance).clamp(-8.0, 8.0)
        y_r = (y_delta / y_tolerance).clamp(-8.0, 8.0)
        b_g = torch.exp(-0.5 * b_r.square())
        y_g = torch.exp(-0.5 * y_r.square())

        row_index = torch.arange(n_cleavages, device=device)[None]
        cleavage_mask = row_index < (peptide_length[:, None] - 1).clamp_min(0)
        cell_mask = cleavage_mask[:, :, None] & top_valid[:, None, :]
        b_g = b_g.masked_fill(~cell_mask, 0.0)
        y_g = y_g.masked_fill(~cell_mask, 0.0)

        # Expose normalized peak intensity directly in the cell features.
        # Peak-selection priority separately multiplies the learned
        # coefficient by g, so distant peaks remain suppressed before argmax.
        intensity_grid = normalized_intensity[:, None, :].expand_as(b_g)
        intensity_grid = intensity_grid.masked_fill(~cell_mask, 0.0)
        b_cell_features = torch.stack([b_g, intensity_grid], dim=1)
        y_cell_features = torch.stack([y_g, intensity_grid], dim=1)

        return b_cell_features, y_cell_features, cell_mask, cleavage_mask

    def prepare_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Build [g, normalized intensity] b/y features on the current device.

        Intensities use the existing ShiftSearch normalization exactly:
        sqrt(non-negative intensity), followed by per-spectrum L1 normalization.
        """
        top_mz, normalized_intensity, top_valid = self.prepare_spectrum_batch(
            batch["mz_array"],
            batch["intensity_array"],
            batch["raw_peak_mask"],
        )
        (
            b_cell_features,
            y_cell_features,
            cell_mask,
            cleavage_mask,
        ) = self.build_candidate_features(
            top_mz,
            normalized_intensity,
            top_valid,
            batch["prefix_masses"],
            batch["peptide_length"],
        )

        batch["b_cell_features"] = b_cell_features
        batch["y_cell_features"] = y_cell_features
        batch["cell_mask"] = cell_mask
        batch["cleavage_mask"] = cleavage_mask
        batch["npeaks"] = top_valid.sum(dim=1).to(torch.long)
        return batch

    def forward(
        self,
        b_cell_features: torch.Tensor,
        y_cell_features: torch.Tensor,
        cell_mask: torch.Tensor,
        cleavage_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(b_cell_features, y_cell_features, cell_mask, cleavage_mask)

    def on_train_epoch_start(self) -> None:
        train_loader = self.trainer.train_dataloader
        sampler = getattr(train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)

    def _shared_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        logits = self(
            batch["b_cell_features"],
            batch["y_cell_features"],
            batch["cell_mask"],
            batch["cleavage_mask"],
        )
        labels = batch["label"].view(-1)
        return self.loss_fn(logits, labels)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        batch = self.prepare_batch(batch)
        metrics = self._shared_step(batch)
        batch_size = batch["label"].shape[0]
        self._log_metrics(
            "train_epoch",
            metrics,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            "train/loss",
            metrics["loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
            batch_size=batch_size,
        )
        if self.log_interval > 0 and (batch_idx + 1) % self.log_interval == 0:
            lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
            self._log_metrics(
                "train_step",
                metrics,
                batch_size=batch_size,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
            self.log(
                "train_step/lr",
                lr,
                on_step=True,
                on_epoch=False,
                logger=True,
                batch_size=batch_size,
            )
        return metrics["loss"]

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        batch = self.prepare_batch(batch)
        metrics = self._shared_step(batch)
        batch_size = batch["label"].shape[0]
        self._log_metrics(
            "val_epoch",
            metrics,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/loss",
            metrics["loss"],
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return metrics["loss"]

    def on_train_epoch_end(self) -> None:
        self._print_epoch_summary("train", "train_epoch")

    def on_validation_epoch_end(self) -> None:
        self._print_epoch_summary("val", "val_epoch")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        total_steps = int(self.trainer.estimated_stepping_batches)
        steps_per_epoch = max(1, total_steps // max(self.n_epochs, 1))

        if self.warmup_steps_cfg is not None and self.warmup_epochs not in (None, 0):
            raise ValueError("Set only one of warmup_steps or warmup_epochs")
        if self.warmup_steps_cfg is None:
            warmup_steps = (
                0
                if self.warmup_epochs in (None, 0)
                else int(self.warmup_epochs) * steps_per_epoch
            )
        else:
            warmup_steps = int(self.warmup_steps_cfg)

        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_steps,
            warmup_steps,
            self.min_lr_ratio,
        )
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _log_metrics(
        self,
        prefix: str,
        metrics: Dict[str, torch.Tensor],
        batch_size: int,
        on_step: bool,
        on_epoch: bool,
        sync_dist: bool,
    ) -> None:
        for name, value in metrics.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                logger=True,
                sync_dist=sync_dist,
                batch_size=batch_size,
            )

    def _print_epoch_summary(self, label: str, prefix: str) -> None:
        metrics = self.trainer.callback_metrics
        summary = {}
        for name in ("loss", "accuracy", "precision", "recall"):
            key = f"{prefix}/{name}"
            if key in metrics:
                summary[name] = float(metrics[key].detach().item())
        if label == "train":
            summary["lr"] = float(self.trainer.optimizers[0].param_groups[0]["lr"])
        if summary:
            self.print(f"epoch={self.current_epoch} {label}={summary}")


__all__ = ["PSMHierarchicalBY"]
