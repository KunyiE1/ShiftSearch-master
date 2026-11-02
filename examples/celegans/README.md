# Real C. elegans search example

This directory contains 100 experimental spectra and a small target–decoy
peptide database. Use the command in the [repository README](../../README.md)
to search them with the included pretrained model.

## Files

- `spectra.mgf`: 100 real spectra, with their original headers and peak values.
- `peptides.fasta`: 8,280 peptide entries preserving all mass-compatible
  candidates for these spectra under the example search settings.
- `search.yaml`: the search configuration, including modifications and mass tolerance.
- `expected_best.tsv`: 94 reference best-hit assignments from the original search.
- `manifest.json`: source information, extraction details and file hashes.

Six spectra have no mass-compatible candidate. The reference assignments are
search results, not ground-truth annotations. With `fdr: 1`, the example retains
both target and decoy best hits without 1% FDR filtering. It demonstrates
execution and must not be used to estimate full-dataset identification performance.

The spectra are selected at uniformly spaced positions in the source MGF,
without using scores or peptide assignments. The small peptide database is
intended for these spectra and settings, not for searching other datasets.

## Sources

Spectra come from the *C. elegans* dataset
[PXD001364 in PRIDE](https://www.ebi.ac.uk/pride/archive/projects/PXD001364).
The source MGF is `C.elegans_PXD001364.filtered_minpeak64.mgf`.
The peptide entries come from `Cele_target_decoy_peptide.fasta`, prepared from
[UniProt proteome UP000001940](https://www.uniprot.org/proteomes/UP000001940)
using the protein-database snapshot dated 2026-08-13.

Consult the source records for attribution and applicable data terms. The
repository's source-code license does not assign a new license to these data.
