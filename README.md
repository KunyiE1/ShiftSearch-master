# ShiftSearch

ShiftSearch learns a score for a spectrum and a candidate peptide by organizing
peak evidence along peptide cleavages. This repository contains the model used
in the manuscript, pretrained inference weights, database search, training and
data-preparation utilities. The software release version is **1.0.0**.

The model uses Gaussian-kernel mass proximity and normalized peak intensity,
selects one representative peak for each singly charged b/y ion at each
cleavage, and combines neighboring cleavage evidence with a 1D CNN. It has
**11,825 trainable parameters** and does not require a peptide-embedding database.
See the [model card](docs/model_card.md) for details.

## Install in a clean environment

Run the following commands from this repository's root. First select a
**Python 3.10** interpreter: `python --version` must report `3.10.x`.
For example, activate an existing Python 3.10 Conda environment or use a
Python 3.10 installation. Only the interpreter is reused when creating the
venv below; its existing third-party packages are not inherited.

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

After activation, use the same commands on either platform:

```bash
python -m pip install --upgrade pip
```

Choose **one** PyTorch build:

```bash
# CPU-only
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

```bash
# Alternatively, NVIDIA GPU with a compatible driver: CUDA 12.1 wheel
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

These wheel commands follow the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/#v240).
Install the remaining pinned dependencies and the package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
shiftsearch-search --help
shiftsearch-train --help
shiftsearch-fdr --help
```

Pip also installs the dependencies required by the listed packages. The default
training configuration uses CSV logging; W&B is optional and disabled. Python
3.10, PyTorch 2.4.0 and Lightning 2.5.0 are the reference runtime.

The console commands are also available as `python -m ShiftSearch.cli.search`,
`python -m ShiftSearch.cli.train` and `python -m ShiftSearch.cli.fdr`. If
PowerShell blocks environment activation, use `.venv\Scripts\python.exe` in
place of `python` and these module forms.

## Run the included real-data example

The repository includes 100 experimental *C. elegans* spectra from
[PXD001364](https://www.ebi.ac.uk/pride/archive/projects/PXD001364) and a small
target–decoy peptide database. Run the pretrained model on these files without
downloading the full dataset or using a GPU:

```bash
shiftsearch-search --mgf examples/celegans/spectra.mgf --peptide-fasta examples/celegans/peptides.fasta --checkpoint weights/shiftsearch.ckpt --config configs/model.yaml --search-config examples/celegans/search.yaml --output outputs/celegans_example/results.tsv --device cpu --batch-size 64
python scripts/check_search_example.py --results outputs/celegans_example/results.best.tsv --expected examples/celegans/expected_best.tsv
python -m unittest discover -s tests -v
```

The example produces 94 best-hit rows; six spectra have no mass-compatible
candidate. It preserves the complete ordered candidate list of the original
database for every included spectrum. The reference table contains search
assignments, rather than known ground-truth annotations. The checker requires
identical peptide assignments and target/decoy labels. The CPU run preserves
all 94 reference assignments; its maximum absolute score difference from the
original GPU run is approximately `1.69e-4`. The checker allows `5e-4` for
CPU/GPU numerical variation, while the reference table retains the original
full-search scores.

To use an NVIDIA GPU, run the same example with a separate output path:

```bash
shiftsearch-search --mgf examples/celegans/spectra.mgf --peptide-fasta examples/celegans/peptides.fasta --checkpoint weights/shiftsearch.ckpt --config configs/model.yaml --search-config examples/celegans/search.yaml --output outputs/celegans_cuda/results.tsv --device cuda --batch-size 64
python scripts/check_search_example.py --results outputs/celegans_cuda/results.best.tsv --expected examples/celegans/expected_best.tsv
```

Its configuration uses `fdr: 1` and retains all best hits; this small subset is
an execution example, not an FDR-performance benchmark. Choose a new output
path when repeating search. See the [example description](examples/celegans/README.md)
for source data, candidate preservation and rebuilding instructions.

The [validation record](docs/validation.md) documents clean-environment
installation and the example/test outcomes, including their platform limits.

## Search experimental spectra

Inputs are a **centroided MGF** with precursor m/z and charge, and a
**pre-digested target–decoy peptide FASTA**. FASTA entries contain peptide
backbone sequences; modifications are specified in the search configuration.
IDs beginning with `DECOY_` identify decoys. The search command does not digest
a protein FASTA or infer missing precursor charge.

To apply the manuscript's spectrum-selection criteria:

```bash
python scripts/filter_mgfs.py --input data/input.mgf --output data/input.filtered.mgf --min-peaks 64 --min-charge 2 --max-charge 4
```

Search with the included pretrained weights:

```bash
shiftsearch-search --mgf data/input.filtered.mgf --peptide-fasta data/peptides.target_decoy.fasta --checkpoint weights/shiftsearch.ckpt --config configs/model.yaml --search-config configs/search_oxi_max2.yaml --output outputs/search/results.tsv --device cuda --batch-size 1024 --spectra-per-batch 32
```

For CPU inference, use `--device cpu`; a smaller `--batch-size`, such as 64,
is useful for a small CPU test. `--batch-size` controls candidate PSM pairs
per forward pass. `--spectra-per-batch` controls spectra preprocessed together.
Use explicit config, weight and data paths when calling the installed package
from another working directory.

| Search configuration | Fixed modification | Variable modifications | Maximum variable modifications |
|---|---|---|---:|
| `configs/search_oxi_max2.yaml` | C +57.021 | M +15.995 | 2 |
| `configs/search_9species.yaml` | C +57.021 | M +15.995 | 3 |
| `configs/search_hela_phospho_max2.yaml` | C +57.021 | M +15.995; S/T/Y +79.966 | 2 total |

All three use a 10 ppm precursor window, peptide lengths 7–32 and float32
search (`amp: false`). The nine-species configuration uses `fdr: 1` to retain
all reported top-1 assignments. Reproducing that accuracy experiment also
requires its sampled spectra and augmented peptide databases.

### Results and FDR

- `results.tsv` contains retained top-k candidates, not every scored candidate.
- `results.best.tsv` contains one winning candidate per spectrum with a hit,
  including both target and decoy assignments. Spectra without a hit are absent.
- `results.fdr_01.best.tsv` is the search helper's convenience output when
  the configured FDR threshold is 0.01.

For the manuscript's FDR procedure, recalculate from the complete
`results.best.tsv`:

```bash
shiftsearch-fdr --input outputs/search/results.best.tsv --output outputs/search/paper.fdr_01.targets.tsv --all-output outputs/search/paper.q_values.tsv --fdr 0.01
```

This groups equal scores at the same threshold, calculates D/T without a
pseudocount, clips threshold FDR to one and applies reverse cumulative minima
to obtain q-values. The accepted-output file contains target PSMs only.
The search helper retains its original row-by-row q-value calculation; use the
separate FDR command for manuscript-consistent results. Supply all top-1 target
and decoy rows, rather than a prefiltered result or a top-k table.

Identification rates divide accepted target PSMs by **all input spectra**.
Peptide sequences ignore modification states; peptidoforms retain modification
types and sites, including an unmodified state. These identities are collected
from accepted target PSMs; peptide-level FDR is not separately controlled.

## Train

Training accepts annotated MGF or spectrum LMDB files and prepared positive/
negative pair TSVs or LMDB caches. See [data preparation](docs/data_preparation.md)
for input formats and cache creation.

```bash
shiftsearch-train --data data/MassiveKB_v2_HCD.lmdb --train-pairs data/MassiveKB_v2_HCD_training_pairs.train.lmdb --val-pairs data/MassiveKB_v2_HCD_training_pairs.val.lmdb --config configs/model.yaml --devices 1 --n-workers 40 --n-epochs 50
```

The recorded configuration uses one GPU, batch size 1024, no gradient
accumulation, one warmup epoch and a maximum of 50 epochs. Early stopping
monitors validation accuracy with patience five. Forty workers is the cloud-run
setting; reduce it to suit local CPU and RAM resources.

With `configs/model.yaml`, checkpoints are saved in `checkpoints_shiftsearch/`.
A relative checkpoint directory is resolved against the model configuration's
grandparent directory. To check one batch, add `--smoke-test --batch-size 2
--n-workers 0`. Scientific retraining requires the prepared pair tables and
spectrum-level splits used in the manuscript, which are external data.

`weights/shiftsearch.ckpt` contains the selected model's exact tensors and is
for inference. It omits optimizer and callback state. Resume your own training
with a full training checkpoint using `--resume path/to/last.ckpt`.

### Optional synthetic training smoke test

To exercise training without downloading the training dataset:

```bash
python examples/create_tiny_demo.py --output-dir demo_work
shiftsearch-train --data demo_work/spectra.mgf --train-pairs demo_work/train_pairs.tsv --val-pairs demo_work/val_pairs.tsv --config demo_work/model.yaml --devices 1 --n-workers 0 --n-epochs 1
```

The generated configuration selects CPU training and performs two optimizer
steps. Its artificial spectra and overlapping training/validation fixtures
test execution only; their metrics are not scientific results. This is separate
from the included real-data search example. See [examples](examples/README.md).

## License and citation

The inherited AGPL-3.0 license and attribution are retained in [LICENSE](LICENSE)
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Experimental datasets have
their own access terms. Citation metadata is in [CITATION.cff](CITATION.cff);
add the manuscript's final bibliographic record when available.

Before public release, complete the [release checklist](RELEASE_CHECKLIST.md).
