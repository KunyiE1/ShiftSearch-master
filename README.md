# ShiftSearch

ShiftSearch scores spectrum–peptide matches using peak evidence organized along
peptide cleavages. This repository includes the pretrained model, search and
training code, and a small example using real experimental spectra.

Run all commands below from the repository root.

## 1. Environment setup

Use **Python 3.10** and create a new environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Upgrade pip, then choose **one** PyTorch installation:

```bash
python -m pip install --upgrade pip
```

For an NVIDIA GPU with a CUDA 12.1-compatible driver:

```bash
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

Or, for CPU-only use:

```bash
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining dependencies and ShiftSearch:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 2. Database search

The included example contains 100 real *C. elegans* spectra, a small target–decoy
peptide database, and the pretrained model. No additional data download is needed.

```bash
shiftsearch-search --mgf examples/celegans/spectra.mgf --peptide-fasta examples/celegans/peptides.fasta --checkpoint weights/shiftsearch.ckpt --config configs/model.yaml --search-config examples/celegans/search.yaml --output outputs/celegans_example/results.tsv --device cpu --batch-size 64
```

For GPU search, change `--device cpu` to `--device cuda`. Choose a new output
path when repeating a search; existing output files are not overwritten.

The output directory contains:

- `results.tsv`: the top-ranked candidates for each spectrum, up to 10.
- `results.best.tsv`: the highest-scoring candidate for each spectrum with a hit,
  including target and decoy assignments.

The example produces 94 best hits; six spectra have no mass-compatible candidate.
Its configuration uses `fdr: 1`, so these outputs are not filtered at 1% FDR.
This small example demonstrates search execution, not full-dataset performance.
Source information is provided in the [example description](examples/celegans/README.md).

To search your own data, replace the MGF, peptide FASTA and output paths, and
select a search configuration from `configs/`. The MGF must contain precursor
m/z and charge. The FASTA must contain **peptide sequences**, not undigested
proteins; decoy identifiers begin with `DECOY_`. Modifications are specified in
the search configuration. The example peptide database is intended only for
the supplied spectra.

## 3. Model training

Prepare an annotated spectrum dataset and positive/negative spectrum–peptide
pairs, with separate training and validation sets. The training data are not
included in the repository. Input formats and preparation commands are described
in [data preparation](docs/data_preparation.md).

For prepared LMDB files:

```bash
shiftsearch-train --data data/MassiveKB_v2_HCD.lmdb --train-pairs data/MassiveKB_v2_HCD_training_pairs.train.lmdb --val-pairs data/MassiveKB_v2_HCD_training_pairs.val.lmdb --config configs/model.yaml --devices 1 --n-workers 40 --n-epochs 50
```

Training settings are in `configs/model.yaml`. The supplied configuration uses
one GPU, batch size 1024 and a maximum of 50 epochs, with early stopping based
on validation accuracy. Adjust `--n-workers` and `--batch-size` for your hardware.
Checkpoints are saved in `checkpoints_shiftsearch/`.

The included `weights/shiftsearch.ckpt` is for inference only. To resume your
own training, add `--resume path/to/last.ckpt` using a full checkpoint generated
during training.

---

Code is distributed under [AGPL-3.0](LICENSE); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution. Experimental
data have their own access terms. Citation metadata is in [CITATION.cff](CITATION.cff).
