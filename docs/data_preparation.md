# Preparing input data

Run the commands below from the repository root after installing its dependencies.
All paths are examples; the original large datasets and result files are not
included in this code repository.

## Spectra used for evaluation

Convert the source raw files to centroided MGF using a suitable conversion tool.
Keep the original experimental peak m/z and intensity values and the precursor
m/z and charge. ShiftSearch does not infer missing precursor charges.

The following utility applies the study's general evaluation criteria: at least
64 peaks and precursor charge 2–4, inclusive. It combines the input files into a
new MGF and writes the counts for each source file to a summary TSV.

```bash
python scripts/filter_mgfs.py \
  --input data/run_01.mgf data/run_02.mgf \
  --output data/evaluation.filtered.mgf \
  --min-peaks 64 --min-charge 2 --max-charge 4
```

Alternatively, use `--input-dir data/original_mgfs` instead of `--input`. This
selects every `.mgf` directly inside that directory; keep previously generated
outputs elsewhere. Filtering preserves spectrum titles and retains all peaks
within each selected spectrum. The model later selects up to 256 peaks and
normalizes intensities as part of model input construction. These are different
steps from deciding which spectra enter the evaluation.

## Search peptide databases

Search takes a **peptide FASTA**, not a protein FASTA. The FASTA sequence lines
contain unmodified amino acid backbones; modifications are enumerated during
search using the search YAML. A header beginning with `DECOY_` marks a decoy.
The same backbone may occur under several protein IDs; preserve the source
protein annotations when preparing the database.

The manuscript uses Protein Digestion Simulator v2.4.8937 (PNNL), obtained
separately from its official release:

https://github.com/PNNL-Comp-Mass-Spec/Protein-Digestion-Simulator/releases/tag/v2.4.8937

For the target–decoy experiments, first reverse each complete protein sequence
and then digest targets and reversed proteins with the same enzyme and settings.
The utility below reuses `reverse_protein_decoy` from the search package:

```bash
python scripts/make_target_decoy_proteins.py \
  --fasta data/proteome.fasta \
  --output data/proteome.target_decoy.fasta
```

Digest this protein FASTA in Protein Digestion Simulator, allowing up to two
missed cleavages. Use the enzyme specified for the dataset in the manuscript;
in the large-scale experiments this is Lys-C for *C. elegans* and trypsin for
*A. thaliana*, *E. coli*, HEK293 and HeLa. Keep the target/decoy protein names in
the exported tab-delimited table. The code does not replace the external
digestion software with a newly implemented digestion algorithm.

Convert the resulting table to the peptide FASTA used by the search CLI:

```bash
python scripts/digest_table_to_fasta.py \
  --input data/proteome.target_decoy_digest.txt \
  --output data/peptides.target_decoy.fasta \
  --min-length 7 --max-length 32 --min-mass 500 --max-mass 5000
```

The default PDS table columns are `Protein_Name` and `Sequence`; change
`--protein-column` or `--sequence-column` if the export uses different names.
The length and mass limits are inclusive. Neutral monoisotopic backbone mass
is calculated as the sum of residue masses plus water, using the same mass
constants as the model. Sequence rows containing unsupported residues are
excluded and counted. The converter preserves every retained table row and
its target/decoy designation; it does not reverse individual digested peptides,
change cleavage sites, enumerate modifications, or add new ground-truth
peptides. Use the released peptide FASTA files for exact manuscript benchmarks.

The nine-species search-accuracy databases contain added ground-truth peptide
backbones that are absent after digestion. For exact reproduction, use the
released augmented databases and sampled spectra together. Do not substitute
an arbitrary digestion-only database for those augmented inputs.

## Training from the released pair tables

The training data consist of two linked components:

1. Experimental spectra, in an annotated MGF or a spectrum LMDB cache.
2. Prepared spectrum–peptide pair tables, in TSV or pair LMDB format.

Use the deposited training and validation pair tables to preserve the actual
split assignments, candidate identities and labels. The public pair data are
named `MassiveKB_v2_HCD_training_pairs.train.tsv` and
`MassiveKB_v2_HCD_training_pairs.val.tsv`. A separate `test.tsv` belongs to this
MassIVE-KB pair collection; it is distinct from the nine-species search-accuracy
benchmark. Obtain the deposited data using the Data Availability links supplied
with the manuscript. A final public Figshare DOI is not embedded in this code.

The experimental spectrum collection is derived from the in vivo HCD component
of MassIVE-KB v.2. Its metadata identify the original files, scans and annotations:

https://massive.ucsd.edu/ProteoSAFe/status.jsp?task=adfc82524d394cb9be709fa3b2379bad

Use its **All Candidate Spectra** metadata (`filename`, `scan`, `annotation`) to
recover spectra from the corresponding source mzML/mzXML files. For the spectra
referenced by the released pair tables, the positive pair rows also provide the
source annotations. Preserve the complete source path and native scan identity;
the same scan number can occur in different files. The representative spectral
library download is a separate MassIVE-KB product and should not be assumed to
include all experimental spectra referenced by the pair tables.

### Annotated MGF and pair schema

For each experimental spectrum, the MGF must contain `PEPMASS`, `CHARGE`, a
unique `TITLE` and an original `SEQ` annotation. For this collection titles take
the form `File:{filename}_Scan:{scan}`. Preserve that full string when matching
pair rows to spectra. Use native scan values, not positions in the file.

The production pair reader requires these five TSV columns:

| Column | Meaning |
|---|---|
| `title` | Full spectrum title; used as the primary spectrum-cache key. |
| `filename` | Source filename/path; retained for the file-and-scan fallback. |
| `scan` | Native scan ID, read as a string. |
| `peptide_sequence` | Candidate peptide with signed mass shifts. |
| `label` | `1` for a positive pair or `0` for a negative pair. |

The released tables include additional provenance and support columns. Keep
those columns when archiving the tables; the training reader selects the fields
it needs. If `title` is empty, the lookup key is the combination of `filename`
and `scan`, which must also agree with the spectrum cache.

Modification syntax is plain signed masses, for example `M+15.995`,
`C+57.021` or an N-terminal `+43.006PEPTIDE`. Bracket notation such as
`M[+15.995]` is not accepted by this training pair parser. Mass shifts in the
annotations are already part of the candidate; do not add fixed modifications
a second time. The model configuration accepts training peptide lengths 6–32.

### Build caches using the existing dataset implementation

The cache builder calls the same `PositivePairPSMDataset` and
`PairTablePSMDataset` classes used during training. It does not generate
negatives or reshuffle splits.

```bash
python scripts/build_training_caches.py \
  --data data/MassiveKB_v2_HCD.mgf \
  --train-pairs data/MassiveKB_v2_HCD_training_pairs.train.tsv \
  --val-pairs data/MassiveKB_v2_HCD_training_pairs.val.tsv \
  --config configs/model.yaml \
  --map-size-gb 256
```

This produces a spectrum cache `data/MassiveKB_v2_HCD.lmdb` and pair caches
`data/MassiveKB_v2_HCD_training_pairs.train.lmdb` and
`data/MassiveKB_v2_HCD_training_pairs.val.lmdb`. The GiB option sets the maximum
LMDB map size for newly created caches; allow enough headroom for the dataset.
The original default map is much smaller than the complete MassIVE-KB data.
Cache creation needs disk space and does not require a GPU. The utility checks
the first few and last retained pair in each split against the spectrum cache;
`--check-items` controls the number of initial checks.

Existing caches are reused, including caches downloaded from another machine.
An old source-path metadata warning is expected when a cache moves between
machines. Do not enable cache rebuilding when `--data` already names an LMDB.

### Start training with LMDB caches

```bash
python bin/train.py \
  --data data/MassiveKB_v2_HCD.lmdb \
  --train-pairs data/MassiveKB_v2_HCD_training_pairs.train.lmdb \
  --val-pairs data/MassiveKB_v2_HCD_training_pairs.val.lmdb \
  --config configs/model.yaml \
  --devices 1 --n-workers 40 --n-epochs 50
```

The train CLI can also take the original MGF and the prepared TSV pair tables
directly. It creates missing caches using the dataset code. The separate cache
builder is useful for choosing a larger LMDB map size before a long training
run. For a small first check, add `--smoke-test --batch-size 2 --n-workers 0` to
the training command. See the repository README for the current CLI names and
the model/search examples.

For exact reproduction, use the deposited pair tables and preserve their split
assignments. The supplied cache builder converts those prepared pairs into the
training input format; it does not regenerate negative candidates or create a
new train/validation split.
