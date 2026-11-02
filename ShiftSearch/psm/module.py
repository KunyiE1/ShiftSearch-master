import math
from typing import Dict, Optional

import lightning.pytorch as pl
import torch
from torch import optim

from ShiftSearch.psm.cnn import PSM2DCNN
from ShiftSearch.psm.loss import PSMBinaryLoss
from ShiftSearch.constants import PROTON, H2O


def build_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
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


class PSMBinaryCNN(pl.LightningModule):
    def __init__(self, model_params: Dict, train_params: Dict, meta_params: Dict) -> None:
        super().__init__()
        self.train_params = dict(train_params)
        self.meta_params = meta_params
        self.lr = float(train_params.get("lr", 1e-3))
        self.n_epochs = int(train_params.get("n_epochs", 1))
        self.warmup_steps_cfg = train_params.get("warmup_steps")
        self.warmup_epochs = train_params.get("warmup_epochs")
        self.min_lr_ratio = float(train_params.get("min_lr_ratio", 0.0))
        self.log_interval = int(train_params.get("log_interval", 0))
        self.save_hyperparameters({"model_params": model_params, "train_params": self.train_params})

        # GPU preprocessing params
        self.max_mz = float(meta_params["max_mz"])
        self.max_pep_len = int(meta_params["max_pep_len"])
        self.max_n_peaks = int(meta_params["max_npeaks"])
        self.convert_y_to_b = bool(meta_params.get("convert_y_to_b", True))
        self.norm = True
        self.ppm_clip = float(meta_params.get("ppm_clip", 50.0))
        self.ppm_hidden_dim = int(meta_params.get("ppm_hidden_dim", 16))
        self.ppm_output_dim = int(meta_params.get("ppm_output_dim", 8))

        encoder_params = dict(model_params)
        encoder_params["ppm_hidden_dim"] = self.ppm_hidden_dim
        encoder_params["ppm_output_dim"] = self.ppm_output_dim
        self.encoder = PSM2DCNN(**encoder_params)
        self.loss_fn = PSMBinaryLoss()

    def prepare_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Prepare one raw batch on the current device.

        Expected input tensors:
        - mz_array: [B, N]
        - intensity_array: [B, N]
        - raw_peak_mask: [B, N]
        - prefix_masses: [B, L+1]
        - charge: [B, 1]
        - pre_mz: [B, 1]
        - peptide_length: [B]
        """
        device = batch["label"].device

        mz = batch["mz_array"].to(device)
        intensity = batch["intensity_array"].to(device)
        raw_peak_mask = batch["raw_peak_mask"].to(device)
        prefix_masses = batch["prefix_masses"].to(device)
        charge = batch["charge"].view(-1, 1).to(device=device, dtype=torch.float32)
        pre_mz = batch["pre_mz"].view(-1, 1).to(device=device, dtype=torch.float32)
        peptide_length = batch["peptide_length"].to(device)

        # Compute ppm error early — independent of fragment preprocessing below.
        # prefix_masses[i, peptide_length[i]] is the sum of all L residues+shifts.
        # Adding H2O gives the full neutral peptide mass, matching search-side cand[2].
        spectrum_neutral_mass = pre_mz * charge - charge * PROTON
        theoretical_mass = prefix_masses.gather(1, peptide_length.view(-1, 1)) + H2O
        ppm_error = (theoretical_mass - spectrum_neutral_mass) / spectrum_neutral_mass.clamp_min(1e-8) * 1e6
        ppm_error = ppm_error.clamp(-self.ppm_clip, self.ppm_clip)

        batch["ppm_error"] = ppm_error

        # Raw peak filtering
        valid = raw_peak_mask & (mz < self.max_mz)

        b_like = mz - PROTON
        peak_masses = b_like
        peak_intensities = intensity
        peak_valid = valid

        if self.convert_y_to_b:
            neutral_mass = pre_mz * charge - charge * PROTON
            y_like = neutral_mass - mz + PROTON
            peak_masses = torch.cat([b_like, y_like], dim=1)
            peak_intensities = torch.cat([intensity, intensity], dim=1)
            peak_valid = torch.cat([valid, valid], dim=1)

        peak_valid = peak_valid & (peak_masses > 0) & (peak_masses < self.max_mz)

        # Sort by mass while pushing invalid entries to the end
        inf = torch.full_like(peak_masses, float("inf"))
        sort_keys = torch.where(peak_valid, peak_masses, inf)
        sort_idx = torch.argsort(sort_keys, dim=1)
        peak_masses = torch.gather(peak_masses, 1, sort_idx)
        peak_intensities = torch.gather(peak_intensities, 1, sort_idx)
        peak_valid = torch.gather(peak_valid, 1, sort_idx)

        # Top-k by intensity per spectrum, keep returned peaks in mass order
        masked_intensity = peak_intensities.masked_fill(~peak_valid, float("-inf"))
        k = min(self.max_n_peaks, masked_intensity.shape[1])
        top_idx = torch.topk(masked_intensity, k=k, dim=1).indices
        top_idx, _ = torch.sort(top_idx, dim=1)
        top_masses = torch.gather(peak_masses, 1, top_idx)
        top_intensities = torch.gather(peak_intensities, 1, top_idx)
        top_valid = torch.gather(peak_valid, 1, top_idx)

        if k < self.max_n_peaks:
            pad_cols = self.max_n_peaks - k
            top_masses = torch.cat([top_masses, torch.zeros(top_masses.shape[0], pad_cols, device=device)], dim=1)
            top_intensities = torch.cat([top_intensities, torch.zeros(top_intensities.shape[0], pad_cols, device=device)], dim=1)
            top_valid = torch.cat([top_valid, torch.zeros(top_valid.shape[0], pad_cols, device=device, dtype=torch.bool)], dim=1)

        if self.norm:
            top_intensities = torch.sqrt(torch.clamp(top_intensities, min=0.0))
            top_intensities = top_intensities.masked_fill(~top_valid, 0.0)
            denom = top_intensities.sum(dim=1, keepdim=True).clamp_min(1e-8)
            top_intensities = top_intensities / denom
        else:
            top_intensities = top_intensities.masked_fill(~top_valid, 0.0)

        n_rows = self.max_pep_len - 1
        internal_pm = prefix_masses[:, 1 : 1 + n_rows]
        mass_diff = top_masses.unsqueeze(1) - internal_pm.unsqueeze(2)

        row_idx = torch.arange(n_rows, device=device).unsqueeze(0)
        row_valid = row_idx < (peptide_length.view(-1, 1) - 1).clamp_min(0)
        pair_mask = row_valid.unsqueeze(2) & top_valid.unsqueeze(1)

        pair_tensor = torch.zeros(top_masses.shape[0], 3, n_rows, self.max_n_peaks, device=device, dtype=torch.float32)
        pair_tensor[:, 0] = mass_diff.masked_fill(~pair_mask, 0.0)
        pair_tensor[:, 1] = top_intensities.unsqueeze(1).expand(-1, n_rows, -1).masked_fill(~pair_mask, 0.0)
        pair_tensor[:, 2] = pair_mask.to(torch.float32)

        batch["pair_tensor"] = pair_tensor
        batch["pair_mask"] = pair_mask
        batch["npeaks"] = top_valid.sum(dim=1).to(torch.long)
        return batch

    def forward(self, pair_tensor: torch.Tensor, pair_mask: Optional[torch.Tensor] = None, ppm_error: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(pair_tensor, pair_mask, ppm_error)

    def on_train_epoch_start(self) -> None:
        train_loader = self.trainer.train_dataloader
        sampler = getattr(train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        batch = self.prepare_batch(batch)
        metrics = self._shared_step(batch)
        batch_size = batch["pair_tensor"].shape[0]
        self._log_metrics("train_epoch", metrics, batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=False)
        self.log("train/loss", metrics["loss"], on_step=True, on_epoch=False, prog_bar=True, logger=False, batch_size=batch_size)

        if self.log_interval > 0 and (batch_idx + 1) % self.log_interval == 0:
            lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
            self._log_metrics("train_step", metrics, batch_size=batch_size, on_step=True, on_epoch=False, sync_dist=False)
            self.log("train_step/lr", lr, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size)

        return metrics["loss"]

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        batch = self.prepare_batch(batch)
        metrics = self._shared_step(batch)
        batch_size = batch["pair_tensor"].shape[0]
        self._log_metrics("val_epoch", metrics, batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss", metrics["loss"], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
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
            warmup_steps = 0 if self.warmup_epochs in (None, 0) else int(self.warmup_epochs) * steps_per_epoch
        else:
            warmup_steps = int(self.warmup_steps_cfg)

        scheduler = build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps, self.min_lr_ratio)
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

    def _shared_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        logits = self(batch["pair_tensor"], batch["pair_mask"], batch["ppm_error"])
        labels = batch["label"].view(-1)
        return self.loss_fn(logits, labels)

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
                prog_bar=False,
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
