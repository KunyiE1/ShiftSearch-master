import sys
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, RandomSampler, default_collate, Subset
from torch.utils.data.distributed import DistributedSampler

from ShiftSearch.psm.dataset import PairTablePSMDataset, PositivePairPSMDataset
from ShiftSearch.psm.module import PSMBinaryCNN
from ShiftSearch.constants import PROTON, H2O



def collate_batch(batch):
    return default_collate(batch)


class DistributedReplacementSampler(torch.utils.data.Sampler[int]):
    def __init__(self, dataset, num_samples: int, num_replicas: int, rank: int, seed: int = 42) -> None:
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.samples_per_replica = (self.num_samples + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.samples_per_replica * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randint(len(self.dataset), (self.total_size,), generator=generator).tolist()
        start = self.rank * self.samples_per_replica
        end = start + self.samples_per_replica
        return iter(indices[start:end])

    def __len__(self) -> int:
        return self.samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class PSMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_pairs: str,
        val_pairs: Optional[str],
        meta_params: Dict,
        train_params: Dict,
        batch_size: int,
        n_workers: int,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_path = str(Path(data_path).resolve())
        self.train_pairs = str(Path(train_pairs).resolve())
        self.val_pairs = None if val_pairs is None else str(Path(val_pairs).resolve())
        self.meta_params = meta_params
        self.train_params = dict(train_params)
        self.batch_size = int(batch_size)
        self.n_workers = int(n_workers)
        self.seed = int(seed)
        self.cache_path = self.train_params.get("cache_path")
        self.rebuild_cache = bool(self.train_params.get("rebuild_cache", False))
        self.epoch_samples = self.train_params.get("epoch_samples")
        self.val_samples = self.train_params.get("val_samples")
        self.train_dataset = None
        self.val_dataset = None
        self.val_subset = None

    def prepare_data(self) -> None:
        PositivePairPSMDataset(
            self.data_path,
            self.meta_params,
            cache_backend="lmdb",
            cache_path=self.cache_path,
            rebuild_cache=self.rebuild_cache,
        )
        PairTablePSMDataset(
            pair_table_path=self.train_pairs,
            mgf_path=self.data_path,
            config=self.meta_params,
            cache_backend="lmdb",
            cache_path=self.cache_path,
            rebuild_cache=self.rebuild_cache,
        )
        if self.val_pairs is not None:
            PairTablePSMDataset(
                pair_table_path=self.val_pairs,
                mgf_path=self.data_path,
                config=self.meta_params,
                cache_backend="lmdb",
                cache_path=self._resolve_val_cache_path(),
                rebuild_cache=self.rebuild_cache,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self.train_dataset is None:
            self.train_dataset = PairTablePSMDataset(
                pair_table_path=self.train_pairs,
                mgf_path=self.data_path,
                config=self.meta_params,
                cache_backend="lmdb",
                cache_path=self.cache_path,
                rebuild_cache=False,
            )
        if stage in (None, "fit", "validate") and self.val_pairs is not None and self.val_dataset is None:
            self.val_dataset = PairTablePSMDataset(
                pair_table_path=self.val_pairs,
                mgf_path=self.data_path,
                config=self.meta_params,
                cache_backend="lmdb",
                cache_path=self._resolve_val_cache_path(),
                rebuild_cache=False,
            )
            if self.val_samples is not None and self.val_subset is None:
                val_count = min(int(self.val_samples), len(self.val_dataset))
                generator = torch.Generator().manual_seed(self.seed)
                indices = torch.randperm(len(self.val_dataset), generator=generator)[:val_count].tolist()
                self.val_subset = Subset(self.val_dataset, indices)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized")
        sampler = self._build_train_sampler()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.n_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self.val_subset if self.val_subset is not None else self.val_dataset
        if dataset is None:
            raise RuntimeError("val_dataset is not initialized")
        sampler = self._build_val_sampler(dataset)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.n_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def _build_train_sampler(self):
        world_size = self.trainer.world_size if self.trainer is not None else 1
        global_rank = self.trainer.global_rank if self.trainer is not None else 0
        if self.epoch_samples is not None:
            if world_size > 1:
                return DistributedReplacementSampler(
                    self.train_dataset,
                    num_samples=int(self.epoch_samples),
                    num_replicas=world_size,
                    rank=global_rank,
                    seed=self.seed,
                )
            return RandomSampler(self.train_dataset, replacement=True, num_samples=int(self.epoch_samples))
        if world_size > 1:
            return DistributedSampler(self.train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
        return None

    def _build_val_sampler(self, dataset):
        world_size = self.trainer.world_size if self.trainer is not None else 1
        global_rank = self.trainer.global_rank if self.trainer is not None else 0

        if world_size > 1 and dataset is not None:
            return DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return None

    def _resolve_val_cache_path(self):
        if self.cache_path is None:
            return str(Path(self.data_path).resolve().with_suffix(".lmdb"))
        return str(Path(self.cache_path).resolve())


def resolve_checkpoint_dir(config_path: str, checkpoint_dir: Optional[str]) -> Path:
    checkpoint_value = checkpoint_dir or "checkpoints"
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(config_path).resolve().parent.parent / checkpoint_path
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def resolve_devices(devices_value):
    if devices_value in (None, "auto"):
        return devices_value
    if isinstance(devices_value, int):
        return devices_value
    if isinstance(devices_value, str):
        stripped = devices_value.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return devices_value


def run_smoke_test(datamodule: PSMDataModule, model: PSMBinaryCNN) -> None:
    datamodule.setup("fit")

    out_file = open("smoke_test_output.txt", "w", encoding="utf-8")

    def log(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(str(a) for a in args) + end
        out_file.write(text)
        out_file.flush()
        sys.stdout.write(text)
        sys.stdout.flush()

    def log_matrix(title: str, matrix: np.ndarray, value_fmt: str) -> None:
        log(f"\n--- {title} ---")
        col_width = 10
        header = " " * 8 + "".join(f"{c:>{col_width}d}" for c in range(matrix.shape[1]))
        log(header)
        for r in range(matrix.shape[0]):
            log(f"[{r:2d}]    " + "".join(value_fmt.format(v) for v in matrix[r]))

    def recover_processed_peaks(prefix_masses_np: np.ndarray, pair_tensor_np: np.ndarray, pair_mask_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if pair_tensor_np.ndim == 4:
            pair_tensor_np = pair_tensor_np[0]
        valid_rows = int(pair_mask_np.any(axis=1).sum())
        valid_cols = int(pair_mask_np.any(axis=0).sum())
        if valid_rows == 0 or valid_cols == 0:
            return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
        first_prefix = float(prefix_masses_np[1])
        masses = pair_tensor_np[0, 0, :valid_cols] + first_prefix
        intensities = pair_tensor_np[1, 0, :valid_cols]
        return masses.astype(np.float32), intensities.astype(np.float32)

    ds = datamodule.train_dataset
    raw_sample = ds[0]

    log("=" * 80)
    log("Single-sample CPU pre-topk input (__getitem__ output)")
    log("=" * 80)
    log(f"seq: {raw_sample['seq']}")
    log(f"annotated_seq: {raw_sample['annotated_seq']}")
    log(f"charge: {raw_sample['charge'].item()}")
    log(f"pre_mz: {raw_sample['pre_mz'].item()}")
    log(f"peptide_length: {raw_sample['peptide_length'].item()}")
    log(f"scan: {raw_sample['scan']}")
    log(f"title: {raw_sample['title']}")
    log(f"cpu_pre_topk: {ds.cpu_pre_topk}")
    log(f"valid raw peak count after mz filter + CPU pre-topk: {int(raw_sample['raw_peak_mask'].sum().item())}")

    prefix_masses_np = raw_sample["prefix_masses"].detach().cpu().numpy()
    log("\nprefix_masses (from pair LMDB):")
    for i, mass in enumerate(prefix_masses_np):
        log(f"  [{i:2d}] {float(mass):10.4f}")

    raw_mz_np = raw_sample["mz_array"].detach().cpu().numpy()
    raw_intensity_np = raw_sample["intensity_array"].detach().cpu().numpy()
    raw_mask_np = raw_sample["raw_peak_mask"].detach().cpu().numpy().astype(bool)
    log("\nCPU-pretopk mz_array / intensity_array (padded to fixed length):")
    for i, (mz, intensity, is_valid) in enumerate(zip(raw_mz_np, raw_intensity_np, raw_mask_np)):
        if is_valid:
            log(f"  [{i:3d}] mz={float(mz):12.6f}  intensity={float(intensity):12.6f}")

    single_batch = collate_batch([raw_sample])
    single_batch = model.prepare_batch(single_batch)

    single_pair_tensor = single_batch["pair_tensor"].detach().cpu().numpy()
    single_pair_mask = single_batch["pair_mask"].detach().cpu().numpy()
    single_npeaks = int(single_batch["npeaks"][0].item())
    single_valid_rows = int(single_batch["pair_mask"][0].any(dim=1).sum().item())
    single_valid_cols = int(single_batch["pair_mask"][0].any(dim=0).sum().item())
    proc_masses, proc_intensities = recover_processed_peaks(prefix_masses_np, single_pair_tensor, single_pair_mask[0])

    single_charge = float(raw_sample["charge"].item())
    single_pre_mz = float(raw_sample["pre_mz"].item())
    single_neutral_mass = single_pre_mz * single_charge - single_charge * PROTON
    single_pep_len = int(raw_sample["peptide_length"].item())
    single_theoretical_mass = float(prefix_masses_np[single_pep_len]) + H2O
    single_ppm = (single_theoretical_mass - single_neutral_mass) / single_neutral_mass * 1e6

    log("\n" + "=" * 80)
    log("Single-sample after prepare_batch (GPU-side final preprocessing)")
    log("=" * 80)
    log(f"pair_tensor shape: {tuple(single_batch['pair_tensor'].shape)}")
    log(f"pair_mask shape: {tuple(single_batch['pair_mask'].shape)}")
    log(f"npeaks: {single_npeaks}")
    log(f"valid_rows: {single_valid_rows}")
    log(f"valid_cols: {single_valid_cols}")
    log(f"pre_mz (pepmass): {single_pre_mz:.6f}  charge: {int(single_charge)}  neutral_mass: {single_neutral_mass:.6f}")
    log(f"theoretical_mass (prefix + H2O): {single_theoretical_mass:.6f}")
    log(f"ppm_error: {single_ppm:.4f}  (clamped: {float(single_batch['ppm_error'][0].item()):.4f})")

    log("\nfinal processed peaks after GPU preprocessing/final top-k/normalization:")
    for i, (mass, intensity) in enumerate(zip(proc_masses, proc_intensities)):
        log(f"  [{i:3d}] mass={float(mass):12.6f}  intensity={float(intensity):12.6f}")

    log_matrix("single sample channel 0: mass_diff", single_pair_tensor[0, 0], "{:10.4f}")
    log_matrix("single sample channel 1: intensity", single_pair_tensor[0, 1], "{:10.6f}")
    log_matrix("single sample channel 2: mask_flag", single_pair_tensor[0, 2], "{:10.0f}")
    log_matrix("single sample pair_mask", single_pair_mask[0].astype(np.int32), "{:10d}")

    model.eval()
    with torch.no_grad():
        single_logits = model(single_batch["pair_tensor"], single_batch["pair_mask"], single_batch["ppm_error"])
    log(f"\nsingle-sample logits: {single_logits.detach().cpu().tolist()}")

    batch = next(iter(datamodule.train_dataloader()))
    batch = model.prepare_batch(batch)

    batch_pair_tensor = batch["pair_tensor"].detach().cpu().numpy()
    batch_pair_mask = batch["pair_mask"].detach().cpu().numpy()
    batch_npeaks = batch["npeaks"].detach().cpu().tolist()
    batch_pep_len = batch["peptide_length"].detach().cpu().tolist()

    batch_ppm = batch["ppm_error"].detach().cpu().numpy()
    with torch.no_grad():
        batch_logits = model(batch["pair_tensor"], batch["pair_mask"], batch["ppm_error"])

    log("\n" + "=" * 80)
    log("Full batch after prepare_batch (GPU-side final preprocessing)")
    log("=" * 80)
    log(f"batch_size: {batch_pair_tensor.shape[0]}")
    log(f"pair_tensor shape: {tuple(batch_pair_tensor.shape)}")
    log(f"pair_mask shape: {tuple(batch_pair_mask.shape)}")
    log(f"logits shape: {tuple(batch_logits.shape)}")

    log(f"\n{'idx':>3}  {'seq':<32}  {'pep_len':>7}  {'npeaks':>6}  {'charge':>6}  {'label':>5}  {'ppm_error':>10}  {'logit':>10}")
    log("-" * 100)
    for i in range(batch_pair_tensor.shape[0]):
        log(
            f"{i:3d}  {batch['seq'][i]:<32}  {int(batch_pep_len[i]):7d}  {int(batch_npeaks[i]):6d}  "
            f"{int(batch['charge'][i].item()):6d}  {float(batch['label'][i].item()):5.1f}  {float(batch_ppm[i]):10.4f}  {float(batch_logits[i].item()):10.4f}"
        )

    max_rows = batch_pair_tensor.shape[2]
    max_cols = batch_pair_tensor.shape[3]
    padded_idx = 0
    for i in range(batch_pair_tensor.shape[0]):
        valid_rows = int(batch["pair_mask"][i].any(dim=1).sum().item())
        valid_cols = int(batch["pair_mask"][i].any(dim=0).sum().item())
        if valid_rows < max_rows or valid_cols < max_cols:
            padded_idx = i
            break

    padded_valid_rows = int(batch["pair_mask"][padded_idx].any(dim=1).sum().item())
    padded_valid_cols = int(batch["pair_mask"][padded_idx].any(dim=0).sum().item())
    padded_prefix_masses = batch["prefix_masses"][padded_idx].detach().cpu().numpy()
    padded_proc_masses, padded_proc_intensities = recover_processed_peaks(
        padded_prefix_masses,
        batch_pair_tensor[padded_idx : padded_idx + 1],
        batch_pair_mask[padded_idx],
    )

    log("\n" + "=" * 80)
    log(f"One padded sample in batch (idx={padded_idx})")
    log("=" * 80)
    log(f"seq: {batch['seq'][padded_idx]}")
    log(f"peptide_length: {batch_pep_len[padded_idx]}")
    log(f"npeaks: {batch_npeaks[padded_idx]}")
    log(f"valid_rows: {padded_valid_rows}")
    log(f"valid_cols: {padded_valid_cols}")

    log("\nprefix_masses for padded sample:")
    for i, mass in enumerate(padded_prefix_masses):
        log(f"  [{i:2d}] {float(mass):10.4f}")

    log("\nprocessed peaks for padded sample:")
    for i, (mass, intensity) in enumerate(zip(padded_proc_masses, padded_proc_intensities)):
        log(f"  [{i:3d}] mass={float(mass):12.6f}  intensity={float(intensity):12.6f}")

    log_matrix("padded sample channel 0: mass_diff", batch_pair_tensor[padded_idx, 0], "{:10.4f}")
    log_matrix("padded sample channel 1: intensity", batch_pair_tensor[padded_idx, 1], "{:10.6f}")
    log_matrix("padded sample channel 2: mask_flag", batch_pair_tensor[padded_idx, 2], "{:10.0f}")
    log_matrix("padded sample pair_mask", batch_pair_mask[padded_idx].astype(np.int32), "{:10d}")

    log("\nOutput written to: smoke_test_output.txt")
    out_file.close()


def load_model_weights(model: PSMBinaryCNN, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)


def create_model(model_params: Dict, train_params: Dict, meta_params: Dict) -> PSMBinaryCNN:
    return PSMBinaryCNN(model_params=model_params, train_params=train_params, meta_params=meta_params)


def create_datamodule(
    data_path: str,
    train_pairs: str,
    val_pairs: Optional[str],
    meta_params: Dict,
    train_params: Dict,
) -> PSMDataModule:
    return PSMDataModule(
        data_path=data_path,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        meta_params=meta_params,
        train_params=train_params,
        batch_size=int(train_params.get("batch_size", 16)),
        n_workers=int(train_params.get("n_workers", 0)),
        seed=int(train_params.get("seed", 42)),
    )


def create_callbacks(train_params: Dict, checkpoint_dir: Path):
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best-epoch={epoch:02d}-step={step}-val_acc={val_epoch/accuracy:.6f}",
            monitor="val_epoch/accuracy",
            mode="max",
            save_top_k=int(train_params.get("keep_best_checkpoints", 3)),
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]

    save_every_n_steps = int(train_params.get("save_every_n_steps", 0))
    keep_step_checkpoints = int(train_params.get("keep_step_checkpoints", 5))
    if save_every_n_steps > 0 and keep_step_checkpoints > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="step-{step}",
                monitor="step",
                mode="max",
                save_top_k=keep_step_checkpoints,
                every_n_train_steps=save_every_n_steps,
                auto_insert_metric_name=False,
            )
        )

    if train_params.get("early_stopping", False):
        callbacks.append(
            EarlyStopping(
                monitor="val_epoch/accuracy",
                mode="max",
                patience=int(train_params.get("early_stopping_patience", 5)),
                min_delta=float(train_params.get("early_stopping_min_delta", 0.0)),
            )
        )

    return callbacks


def create_logger(train_params: Dict, checkpoint_dir: Path):
    loggers = []
    local_logger = train_params.get("local_logger", "csv")
    run_name = train_params.get("wandb_run_name") or train_params.get("run_name") or "psm_cnn"
    project_name = train_params.get("project_name", "ShiftSearch")
    wandb_run_id = train_params.get("wandb_run_id")
    log_root = checkpoint_dir.parent / "logs"

    if local_logger == "csv":
        loggers.append(CSVLogger(save_dir=str(log_root), name=run_name))
    if train_params.get("wandb", False):
        wandb_kwargs = {
            "project": train_params.get("wandb_project", project_name),
            "name": train_params.get("wandb_run_name") or run_name,
            "save_dir": str(log_root),
        }
        if wandb_run_id not in (None, "", "null"):
            wandb_kwargs["id"] = str(wandb_run_id)
            wandb_kwargs["resume"] = "allow"
        loggers.append(WandbLogger(**wandb_kwargs))

    if not loggers:
        return False
    if len(loggers) == 1:
        return loggers[0]
    return loggers


def create_trainer(train_params: Dict, checkpoint_dir: Path) -> pl.Trainer:
    return pl.Trainer(
        accelerator=train_params.get("accelerator", "auto"),
        devices=resolve_devices(train_params.get("devices", 1)),
        strategy=train_params.get("strategy", "auto"),
        precision=train_params.get("precision", "16-mixed"),
        max_epochs=int(train_params.get("n_epochs", 1)),
        accumulate_grad_batches=int(train_params.get("accumulate_grad_batches", 1)),
        callbacks=create_callbacks(train_params, checkpoint_dir),
        logger=create_logger(train_params, checkpoint_dir),
        log_every_n_steps=1,
        deterministic=bool(train_params.get("deterministic", False)),
        use_distributed_sampler=False,
        num_sanity_val_steps=0,
    )
