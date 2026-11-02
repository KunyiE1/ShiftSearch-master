# Installation and execution validation

## Clean CPU environment

The README installation and example commands have been executed on Windows 10
with Python 3.10.20 in a newly created `venv`. The initial environment contains
only pip and setuptools. System site packages are excluded and user site
packages are disabled. No existing NumPy, PyTorch, Lightning or other
third-party packages are reused. A pre-existing Python installation provides
the interpreter and standard library only.

The installation follows the README sequence:

1. Create and activate a Python 3.10 virtual environment.
2. Upgrade pip and install the PyTorch 2.4.0 CPU wheel from the official index.
3. Install `requirements.txt`, then install this package with `pip install -e .`.
4. Run `pip check` and all three command help pages.
5. Search the bundled real C. elegans spectra with the manuscript weights.
6. Run the result checker and the complete test suite.

All these steps pass. `pip check` reports no broken requirements. The real
example searches 100 spectra and 8,633 candidate PSMs, producing 915 top-k rows
and 94 best hits; six spectra have no mass-compatible candidate. All **15 unit
and integration tests pass**. The tests include an optional synthetic
training/checkpoint smoke test; it is separate from the real search example.

The final source was copied into a separate checkout before repeating the
entire installation from an empty environment. The real MGF, peptide FASTA,
reference assignments and provenance are included in `examples/celegans/`;
temporary training checkpoints and generated search outputs are not shipped.
The example is an execution/regression test, not a performance benchmark.

## CUDA environment

A second newly created Python 3.10 environment independently follows the same
README sequence, choosing the official PyTorch 2.4.0 CUDA 12.1 wheel instead.
It also starts with only pip and setuptools, does not inherit system or user
site packages, and passes dependency checking. The final 15-test suite also
passes in this environment. A real-data search with the manuscript weights
runs on an NVIDIA GeForce RTX 3080 using `--device cuda` and produces the same
915 top-k rows and 94 best-hit assignments. GPU model training is not part of
this validation run.

All 94 best-hit peptide sequences, modification states, protein IDs, ranks and
target/decoy labels match the corresponding unfiltered full-database search
records exactly. The largest absolute score difference from those original
GPU reference scores is approximately `1.69e-4` on CPU and `4.58e-6` on GPU.
The result checker allows `5e-4` absolute score variation, while all assignment
identity fields remain exact checks. The reference scores are not rewritten
to match the new runs, and reference assignments are not called ground truth.

All 100 MGF blocks match their original-block checksums. For each spectrum,
the complete ordered candidate list from the reduced FASTA is verified against
the original full database. Its six no-candidate spectra are retained. Example
selection uses uniform source positions, not scores or search assignments.

The resolved dependency versions and outcomes for both environments are
recorded in [validated_environment.json](validated_environment.json). These
are actual fresh installations, not merely package import checks against an
existing dependency environment.

## Model fidelity

The included `weights/shiftsearch.ckpt` contains exactly the model tensors from
the selected manuscript checkpoint. Its export hash and tensor digest are in
[`weights/metadata.json`](../weights/metadata.json). Renaming the software and
checkpoint does not alter the model calculation or learned tensors.
Checksums and syntax-tree comparisons of the preserved runtime modules are
recorded in [model_fidelity.json](model_fidelity.json).

The test suite checks original feature construction, cleavage/peak masks,
selected evidence, reference logits, all trainable gradients, candidate
modifications, target/decoy handling and both legacy and paper FDR behavior.
It also rejects checkpoint export with an incompatible Gaussian proximity
scale. Optimizer/callback state is omitted from the inference checkpoint;
it cannot be used to resume an interrupted training run.

## Boundaries

- Local execution validates Windows. Linux CI is configured but is not
  described as already executed remotely.
- Full scientific retraining and all large experimental searches are not
  rerun during repository preparation.
- Prepared training pairs and exact evaluation inputs remain external data;
  see [data preparation](data_preparation.md) and the
  [release checklist](../RELEASE_CHECKLIST.md).
- Internet access is required to download the dependencies. GPU execution
  additionally requires a compatible NVIDIA driver.
