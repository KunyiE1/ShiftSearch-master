# Runnable examples

## Search real experimental spectra

The primary example uses the included experimental *C. elegans* spectra,
target–decoy peptide FASTA and pretrained ShiftSearch weights. Run from the
repository root after installation:

```bash
shiftsearch-search --mgf examples/celegans/spectra.mgf --peptide-fasta examples/celegans/peptides.fasta --checkpoint weights/shiftsearch.ckpt --config configs/model.yaml --search-config examples/celegans/search.yaml --output outputs/celegans_example/results.tsv --device cpu --batch-size 64
python scripts/check_search_example.py --results outputs/celegans_example/results.best.tsv --expected examples/celegans/expected_best.tsv
```

The example has 100 spectra and produces 94 best-hit rows; six spectra have no
mass-compatible candidates. For GPU execution, change the device to `cuda` and
use a new output path, such as `outputs/celegans_cuda/results.tsv`. The checker
also accepts that run's corresponding `results.best.tsv` file. See the
[data description](celegans/README.md) for provenance, source-data links and
rebuilding instructions.

The recorded assignments are search outputs, not ground-truth annotations.
This example checks reproducible execution and does not estimate identification
accuracy or full-dataset FDR performance.

The checker requires exact peptide assignments and target/decoy labels and
allows an absolute score difference of `5e-4`. The measured CPU comparison
preserves all 94 original assignments, with a maximum absolute score difference
of approximately `1.69e-4` from the original GPU search. Reference scores are
those recorded in the full search.

## Optional synthetic training smoke test

This separate fixture exercises training on a CPU without external training
data:

```bash
python examples/create_tiny_demo.py --output-dir demo_work
shiftsearch-train --data demo_work/spectra.mgf --train-pairs demo_work/train_pairs.tsv --val-pairs demo_work/val_pairs.tsv --config demo_work/model.yaml --devices 1 --n-workers 0 --n-epochs 1
```

The generated configuration explicitly selects CPU and float32 training and
performs two optimizer steps. To test loading its newly trained checkpoint:

```bash
shiftsearch-search --mgf demo_work/spectra.mgf --peptide-fasta demo_work/peptides.fasta --checkpoint demo_work/checkpoints/last.ckpt --config demo_work/model.yaml --search-config demo_work/search.yaml --output demo_work/results.tsv --device cpu
```

The synthetic fixture has three artificial spectra and intentionally overlapping
training/validation pairs. Its scores and metrics are not scientific performance
estimates. Choose a new output directory when repeating either example, and
update subsequent paths accordingly.

Run the regression and CPU pipeline tests with:

```bash
python -m unittest discover -s tests -v
```
