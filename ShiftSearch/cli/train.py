import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml

from ShiftSearch.psm.hierarchical_by_module import PSMHierarchicalBY
from ShiftSearch.psm.runner import create_datamodule, create_trainer, resolve_checkpoint_dir


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the ShiftSearch b/y scorer with [g, I] cell features."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-pairs", required=True)
    parser.add_argument("--val-pairs", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


def move_tensors(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_smoke_test(datamodule, model: PSMHierarchicalBY) -> None:
    datamodule.setup("fit")
    # Do not materialize the production sampler's millions of replacement
    # indices merely to fetch one diagnostic batch.
    datamodule.epoch_samples = datamodule.batch_size
    batch = next(iter(datamodule.train_dataloader()))
    print("[smoke] loaded one real LMDB batch", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).train()
    batch = move_tensors(batch, device)
    print(f"[smoke] moved model and batch to {device}", flush=True)
    batch = model.prepare_batch(batch)
    print("[smoke] built normalized b/y cell features", flush=True)
    metrics = model._shared_step(batch)
    print("[smoke] completed forward and loss", flush=True)
    metrics["loss"].backward()
    print("[smoke] completed backward", flush=True)

    print(f"[smoke] device={device}")
    print(f"[smoke] batch={batch['label'].shape[0]}")
    print(f"[smoke] b_cell_features={tuple(batch['b_cell_features'].shape)}")
    print(f"[smoke] y_cell_features={tuple(batch['y_cell_features'].shape)}")
    print(f"[smoke] cleavage_mask={tuple(batch['cleavage_mask'].shape)}")
    print(f"[smoke] loss={float(metrics['loss'].detach()):.6f}")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is None:
            raise RuntimeError(f"Parameter did not receive a gradient: {name}")
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient: {name}")
    print("[smoke] forward/backward and all gradients are finite")


def main(argv=None) -> None:
    args = parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    meta_params = dict(config["Meta"])
    model_params = dict(config["Model"]["params"])
    train_params = dict(config.get("Train", {}))

    for arg_name, config_name in (
        ("batch_size", "batch_size"),
        ("n_workers", "n_workers"),
        ("n_epochs", "n_epochs"),
        ("lr", "lr"),
        ("devices", "devices"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            train_params[config_name] = value
    if args.resume is not None:
        train_params["resume_from_checkpoint"] = args.resume

    pl.seed_everything(int(train_params.get("seed", 42)), workers=True)
    datamodule = create_datamodule(
        data_path=args.data,
        train_pairs=args.train_pairs,
        val_pairs=args.val_pairs,
        meta_params=meta_params,
        train_params=train_params,
    )
    model = PSMHierarchicalBY(model_params, train_params, meta_params)

    if args.smoke_test:
        run_smoke_test(datamodule, model)
        return

    resume = train_params.get("resume_from_checkpoint")
    if resume and not Path(resume).exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume}")

    checkpoint_dir = resolve_checkpoint_dir(args.config, train_params.get("checkpoint_dir"))
    trainer = create_trainer(train_params, checkpoint_dir)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume)


if __name__ == "__main__":
    main()
