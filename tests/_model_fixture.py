"""Small, deterministic model input shared by the regression checks."""

import hashlib
import numpy as np
import torch

from ShiftSearch.constants import H2O, PROTON
from ShiftSearch.psm.hierarchical_by_module import PSMHierarchicalBY
from ShiftSearch.search.peptide_utils import build_prefix_masses


MODEL = {"cell_hidden_dim": 16, "cell_embedding_dim": 16,
         "cleavage_channels": [32, 64], "kernel_size": 3,
         "head_hidden_dim": 32, "dropout": 0.2}
META = {"max_mz": 6000.0, "max_pep_len": 10, "max_npeaks": 6,
        "fragment_tolerance_ppm": 20.0}
TRAIN = {"lr": 0.00035, "n_epochs": 1}


def fixture():
    torch.set_num_threads(1)
    model = PSMHierarchicalBY(MODEL, TRAIN, META).cpu().eval()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = ((torch.arange(parameter.numel(), dtype=torch.float32) % 29) - 14) / (70 + index)
            parameter.copy_(values.reshape(parameter.shape))
    mz, intensity, prefixes, lengths = [], [], [], []
    for annotation in ("PEPTIK", "AC+57.021DM+15.995PEPK"):
        sequence, prefix = build_prefix_masses(annotation, META["max_pep_len"])
        total = float(prefix[len(sequence)])
        mz.append([(float(prefix[1]) + PROTON) * 1.000002,
                   (float(prefix[2]) + PROTON) * 1.000010,
                   (total - float(prefix[3]) + H2O + PROTON) * 0.999990,
                   111.0, 777.7, 0.0, 6000.0, 333.0])
        intensity.append([100.0, 25.0, 81.0, 16.0, 9.0, 400.0, 900.0, -1.0])
        prefixes.append(prefix)
        lengths.append(len(sequence))
    batch = {"mz_array": torch.tensor(mz, dtype=torch.float32),
             "intensity_array": torch.tensor(intensity, dtype=torch.float32),
             "raw_peak_mask": torch.ones(2, 8, dtype=torch.bool),
             "prefix_masses": torch.tensor(np.stack(prefixes)),
             "peptide_length": torch.tensor(lengths, dtype=torch.long)}
    return model, batch


def snapshot(checkpoint=None):
    model, batch = fixture()
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved.get("state_dict", saved), strict=True)
    with torch.no_grad():
        batch = model.prepare_batch(batch)
        b = model.encoder._select_peak(batch["b_cell_features"], batch["cell_mask"], batch["cleavage_mask"])
        y = model.encoder._select_peak(batch["y_cell_features"], batch["cell_mask"], batch["cleavage_mask"])
        logits = model(batch["b_cell_features"], batch["y_cell_features"], batch["cell_mask"], batch["cleavage_mask"])
    state_hash = hashlib.sha256()
    for name, value in model.state_dict().items():
        state_hash.update(name.encode("utf-8"))
        state_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "state_shapes": {k: list(v.shape) for k, v in model.state_dict().items()},
        "state_tensor_sha256": state_hash.hexdigest(),
        "b_cell_features": batch["b_cell_features"].tolist(),
        "y_cell_features": batch["y_cell_features"].tolist(),
        "cell_mask": batch["cell_mask"].tolist(),
        "cleavage_mask": batch["cleavage_mask"].tolist(),
        "b_selected_evidence": b.tolist(), "y_selected_evidence": y.tolist(),
        "logits": logits.tolist(), "scores": torch.sigmoid(logits).tolist(),
    }
