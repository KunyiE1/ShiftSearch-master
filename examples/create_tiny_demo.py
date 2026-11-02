"""Create a tiny synthetic CPU training/search example.

The artificial spectra and untrained deterministic weights are infrastructure
fixtures only. They do not measure peptide-identification performance.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml

from ShiftSearch.constants import H2O, PROTON
from ShiftSearch.psm.hierarchical_by_module import PSMHierarchicalBY
from ShiftSearch.search.peptide_utils import build_prefix_masses


MODEL_PARAMS = {
    "cell_hidden_dim": 16, "cell_embedding_dim": 16,
    "cleavage_channels": [32, 64], "kernel_size": 3,
    "head_hidden_dim": 32, "dropout": 0.2,
}


def fill_deterministic_weights(model):
    """Version-independent arithmetic weights; not a trained checkpoint."""
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32)
            values = ((values % 29) - 14) / (70 + index)
            parameter.copy_(values.reshape(parameter.shape))


def create_demo(output_dir):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "psm_hierarchical_by",
        "Meta": {"max_charge": 6, "start_mz": 0.0, "max_mz": 6000.0,
                 "max_npeaks": 32, "cpu_pre_topk": 64, "min_npeaks": 10,
                 "max_pep_len": 32, "min_pep_len": 6, "fragment_tolerance_ppm": 20.0},
        "Model": {"params": MODEL_PARAMS},
        "Train": {"batch_size": 2, "n_workers": 0, "accumulate_grad_batches": 1,
                  "n_epochs": 1, "lr": 0.00035, "cache_path": None,
                  "rebuild_cache": False, "epoch_samples": 4, "val_samples": 2,
                  "warmup_steps": 0, "warmup_epochs": 0, "min_lr_ratio": 0.01,
                  "log_interval": 0, "checkpoint_dir": str(output / "checkpoints"),
                  "save_every_n_steps": 0, "keep_step_checkpoints": 0,
                  "keep_best_checkpoints": 1, "accelerator": "cpu", "devices": 1,
                  "strategy": "auto", "precision": "32-true", "early_stopping": False,
                  "seed": 42, "deterministic": True, "local_logger": "csv",
                  "run_name": "synthetic_smoke_only", "wandb": False},
    }
    search_config = {
        "Search": {"precursor_ppm": 10.0, "decoy_prefix": "DECOY_", "fdr": 1.0,
                   "batch_size": 2, "spectra_per_batch": 2, "top_k": 2,
                   "prefix_cache_size": 32, "flush_every": 1, "min_score": 0.0,
                   "max_variable_mods": 1, "min_pep_len": 6, "max_pep_len": 32,
                   "amp": False},
        "Mods": {"fixed": "C:+57.021", "variable": "M:+15.995"},
    }
    annotations = ["AC+57.021DMPEPK", "AC+57.021DM+15.995PEPK", "PEPTIDEK"]
    negatives = ["PEPMDC+57.021AK", "PEPM+15.995DC+57.021AK", "EDITPEPK"]
    fasta = ">SYNTHETIC_target_1\nACDMPEPK\n>DECOY_SYNTHETIC_1\nPEPMDCAK\n>SYNTHETIC_target_2\nPEPTIDEK\n>DECOY_SYNTHETIC_2\nEDITPEPK\n"
    (output / "peptides.fasta").write_text(fasta, encoding="utf-8")
    mgf = []
    pairs = []
    for scan, (annotation, negative) in enumerate(zip(annotations, negatives), 1):
        sequence, prefix = build_prefix_masses(annotation, 32)
        total = float(prefix[len(sequence)])
        neutral = total + H2O
        charge = 2
        title = f"File:synthetic_demo_Scan:{scan}"
        peaks = []
        for cleavage in range(1, len(sequence)):
            peaks.extend([(float(prefix[cleavage]) + PROTON, 100 + cleavage * 17),
                          (total - float(prefix[cleavage]) + H2O + PROTON, 90 + cleavage * 13)])
        peaks.extend([(123.456, 9), (456.789, 11), (789.123, 7)])
        lines = ["BEGIN IONS", f"TITLE={title}", f"SCANS={scan}",
                 f"PEPMASS={(neutral + charge * PROTON) / charge:.9f}",
                 f"CHARGE={charge}+", f"SEQ={annotation}"]
        lines += [f"{mz:.9f} {intensity}" for mz, intensity in sorted(peaks)]
        mgf.extend(lines + ["END IONS", ""])
        for peptide, label in ((annotation, 1), (negative, 0)):
            pairs.append([title, "synthetic_demo", str(scan), peptide, label])
    (output / "spectra.mgf").write_text("\n".join(mgf), encoding="utf-8")
    for name in ("train_pairs.tsv", "val_pairs.tsv"):
        with (output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["title", "filename", "scan", "peptide_sequence", "label"])
            writer.writerows(pairs)
    for name, values in (("model.yaml", config), ("search.yaml", search_config)):
        (output / name).write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    model = PSMHierarchicalBY(MODEL_PARAMS, config["Train"], config["Meta"])
    fill_deterministic_weights(model)
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                "fixture_notice": "SYNTHETIC deterministic untrained weights; not the published checkpoint."},
               output / "synthetic_untrained.ckpt")
    info = {"notice": "Synthetic infrastructure test only; not a performance benchmark.",
            "spectra": 3, "peptide_fasta_entries": 4, "train_pairs": 6, "val_pairs": 6,
            "train_val_overlap": "Intentional smoke fixture; not a scientific validation split.",
            "parameters": sum(p.numel() for p in model.parameters())}
    (output / "SYNTHETIC_FIXTURE.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(create_demo(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
