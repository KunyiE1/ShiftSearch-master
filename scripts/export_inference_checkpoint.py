"""Export model tensors without optimizer/callback state or machine-local paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from ShiftSearch.psm.hierarchical_by_module import PSMHierarchicalBY


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    paths = [args.output.resolve(), args.metadata.resolve()]
    if len(set(paths)) != 2 or args.input.resolve() in paths:
        raise ValueError("Input, output and metadata paths must differ")
    if any(p.exists() for p in paths):
        raise FileExistsError("Refusing to overwrite an existing export")
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    source = torch.load(args.input, map_location="cpu", weights_only=True)
    original_hparams = source.get("hyper_parameters", {})
    for key, supplied in (("model_params", config["Model"]["params"]),
                          ("meta_params", config["Meta"])):
        # Tensor shapes cannot detect a changed Gaussian scale or peak limit.
        if key in original_hparams and original_hparams[key] != supplied:
            raise ValueError(f"Supplied config {key} differs from the source checkpoint; export refused")
    state = source.get("state_dict", source)
    model = PSMHierarchicalBY(config["Model"]["params"], config["Train"], config["Meta"])
    model.load_state_dict(state, strict=True)
    exported = {name: value.detach().cpu().clone() for name, value in state.items()}
    # Preserve model construction metadata without optimizer/callback paths.
    train_keys = ("lr", "n_epochs", "warmup_steps", "warmup_epochs", "min_lr_ratio", "log_interval")
    payload = {
        "state_dict": exported,
        "hyper_parameters": {"model_params": config["Model"]["params"], "meta_params": config["Meta"],
                             "train_params": {k: config["Train"][k] for k in train_keys if k in config["Train"]}},
        "pytorch-lightning_version": str(source.get("pytorch-lightning_version", "2.5.0")),
        "checkpoint_scope": "inference_only; optimizer/callback/training-resume state omitted",
    }
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    reloaded = torch.load(args.output, map_location="cpu", weights_only=True)["state_dict"]
    if set(reloaded) != set(state) or any(not torch.equal(state[k].cpu(), reloaded[k]) for k in state):
        raise RuntimeError("Export changed a model tensor")
    metadata = {
        "model": "ShiftSearch", "scope": "inference_only", "parameters": sum(p.numel() for p in model.parameters()),
        "source_checkpoint_name": args.input.name,
        "source_checkpoint_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source_epoch_zero_indexed": source.get("epoch"), "source_global_step": source.get("global_step"),
        "export_file": args.output.name, "export_bytes": args.output.stat().st_size,
        "export_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "model_tensor_sha256": tensor_digest(exported), "all_source_model_tensors_identical": True,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
