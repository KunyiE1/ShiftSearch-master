# Regression and CPU smoke tests

The suite uses Python's standard `unittest` runner and the installed package
dependencies. It requires no GPU, private dataset or additional test framework.

After following the installation instructions, run from the repository root:

```bash
python -m unittest discover -s tests -v
```

To check an installed package from another working directory, provide the
full path to this repository's `tests` directory with `-s`.

The fixed regression fixtures use small synthetic inputs and deterministic
untrained weights. They cover the parameter schema, Gaussian/intensity
features, masks, selected b/y evidence, logits and sigmoid scores.

The CLI smoke test creates three artificial spectra and six spectrum–peptide
pairs in a temporary directory. It performs two CPU optimizer steps, loads the
resulting checkpoint and searches the small peptide FASTA through the installed
training and search entry points. Its generated files are temporary.

The search helper and manuscript FDR calculation are tested separately. The
former preserves row-by-row q-values, whereas the latter groups equal score
thresholds before calculating target–decoy D/T.

Pretrained-weight checks use `weights/shiftsearch.ckpt` and its metadata.
They verify the file hash, restricted checkpoint loading, the 11,825-parameter
model schema and fixed CPU outputs. The compact checkpoint must contain no
optimizer or callback state.

These tests verify software behavior and model preservation. Their artificial
data do not establish peptide-identification accuracy or biological validity.
