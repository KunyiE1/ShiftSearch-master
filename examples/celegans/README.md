# Real C. elegans search example

This directory contains a small runnable subset of the experimental inputs
used in the manuscript. It uses the pretrained model in `weights/shiftsearch.ckpt`.

## Run and check

From the repository root, after installation:

```bash
shiftsearch-search --mgf examples/celegans/spectra.mgf --peptide-fasta examples/celegans/peptides.fasta --checkpoint weights/shiftsearch.ckpt --config configs/model.yaml --search-config examples/celegans/search.yaml --output outputs/celegans_example/results.tsv --device cpu --batch-size 64
python scripts/check_search_example.py --results outputs/celegans_example/results.best.tsv --expected examples/celegans/expected_best.tsv
```

To run on an NVIDIA GPU, use `--device cuda` and a new output path such as
`outputs/celegans_cuda/results.tsv`. Check that run's corresponding best-hit
file with the same checker. Existing output files are not overwritten.

## Included files

- `spectra.mgf`: 100 experimental spectra copied as original MGF blocks.
- `peptides.fasta`: 8,280 target/decoy peptide entries needed to preserve every
  mass-compatible candidate for these spectra under the configured search.
- `search.yaml`: 10 ppm precursor window, fixed C +57.021, variable M +15.995,
  at most two variable modifications, peptide lengths 7–32, and float32 scoring.
- `expected_best.tsv`: 94 reference best-hit assignments from the original
  search using the full peptide database. Six spectra have no candidates.
- `manifest.json`: source filenames and hashes, spectrum indices, file hashes,
  search settings and candidate-preservation checks.

The 100 spectra are selected at uniformly spaced positions among the 123,197
eligible spectra, including the first and last, and retain their source order.
Eligible spectra have at least 64 peaks and precursor charge 2–4. Selection
does not use scores or reported peptide assignments, and peak values are not
rewritten during extraction.

The reduced FASTA retains source entry order and full headers. It expands to
13,298 modified/unmodified candidate forms and yields 8,633 candidate PSMs across
the 100 spectra. For each selected spectrum, the ordered candidate list is
checked against the full database after the same modification enumeration and
precursor-mass retrieval. This preserves the original candidate competition,
including decoys and tied-score ordering. The guarantee applies to these spectra
and search settings; other queries or modification settings require an
appropriate full database or a rebuilt subset.

## Interpretation

The reference entries are model search assignments, not independent ground-truth
annotations. The checker matches spectrum title and precursor charge, compares
peptide/modification assignments and target/decoy status, and checks available
sequence, protein-ID and rank fields. These identity fields must match exactly.
The CPU comparison preserves all 94 reference assignments and has a maximum
absolute score difference of approximately `1.69e-4` from the original GPU
search. The checker allows an absolute score difference of `5e-4`; the reference
table retains the original full-search scores. Spectrum indices are not used
for matching because subset extraction renumbers them.

The example uses `fdr: 1`, so no FDR filtering is applied. It is intended to
demonstrate execution and verify preserved search behavior. Its identification
counts or any FDR calculated from this small subset must not be interpreted as
the manuscript's full-dataset performance.

## Sources and attribution

The spectrum source is the *C. elegans* dataset
[PXD001364 in PRIDE](https://www.ebi.ac.uk/pride/archive/projects/PXD001364).
The extracted input is `C.elegans_PXD001364.filtered_minpeak64.mgf`.
The peptide subset comes from `Cele_target_decoy_peptide.fasta`, built for the
manuscript from [UniProt proteome UP000001940](https://www.uniprot.org/proteomes/UP000001940)
using the protein-database snapshot dated **2026-08-13**.

Consult these source records for the original data attribution and applicable
terms. The repository's AGPL source-code license does not by itself assign a
new license to the experimental spectra or proteome-derived data.

## Rebuild the input subset

With the full source MGF and peptide FASTA available locally, run:

```bash
python scripts/extract_search_example.py --source-mgf data/C.elegans_PXD001364.filtered_minpeak64.mgf --source-peptide-fasta data/Cele_target_decoy_peptide.fasta --model-config configs/model.yaml --search-config configs/search_oxi_max2.yaml --max-spectra 100 --dataset-accession PXD001364 --output-dir outputs/rebuilt_celegans_example
```

The output directory must not already exist. This rebuilds the spectrum and
FASTA subset, example configuration and manifest; it does not use search scores
to select them. The full source files are external inputs and are not required
to run the already included example.
