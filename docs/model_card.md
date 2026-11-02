# ShiftSearch model card

## Purpose

ShiftSearch release **1.0.0** provides the scoring architecture and pretrained
model used in the manuscript. It scores a spectrum against a candidate peptide
for closed database search and has 11,825 trainable parameters.

## Inputs and computation

- Training peptide lengths are 6–32; the supplied evaluation search
  configurations use lengths 7–32.
- CPU preprocessing retains up to 512 peaks. The model retains up to 256
  highest-intensity valid peaks, square-root transforms their intensities and
  normalizes the transformed intensities to sum to one.
- Each internal cleavage has theoretical singly charged b and complementary
  y ions calculated with the candidate's modification masses.
- Each cleavage–peak cell has features `[g, I]`, where `I` is normalized
  intensity and
  `g = exp(-0.5 * clip(delta_mz / (theoretical_mz * 20e-6), -8, 8)^2)`.
- A shared encoder produces a 16-dimensional embedding. Selection priority is
  `g * sigmoid(linear(embedding))`. Hard argmax selects one valid peak per
  ion series and cleavage. Its embedding and priority are both retained.
- Concatenated b/y evidence has 34 channels. Two kernel-size-three 1D
  convolutions produce 32 then 64 channels along the cleavage axis. Masked mean
  pooling and an MLP produce a logit and sigmoid score. Search disables dropout.

The 20 ppm scale controls Gaussian decay; it is not a hard peak-admission
window. A representative peak is selected even without a close b/y match.
The model does not explicitly enumerate doubly charged or neutral-loss
channels. A neural score is not a calibrated posterior probability.

## Training and weights

Training uses prepared positive/negative spectrum–peptide pairs derived from
MassIVE-KB v.2 in vivo HCD spectra. Pair labels and spectrum-level splits are
external data. The recorded settings use Adam, learning rate 0.00035,
3,000,000 sampled pairs per epoch, 100,000 validation pairs, batch size 1024,
one warmup epoch, cosine decay and a maximum of 50 epochs. Early stopping
monitors validation accuracy with patience five.

The selected checkpoint is epoch 15 (zero-indexed), global step 46,880,
with recorded validation accuracy 0.970270. The maximum epoch setting does not
mean the selected model completes 50 epochs.

`weights/shiftsearch.ckpt` preserves the selected model's tensors and contains
only inference metadata. `weights/metadata.json` records its hash. Optimizer
and callback state are omitted, so training resumption requires a full
checkpoint produced during training.

## Evaluation scope

The manuscript uses spectra containing at least 64 peaks with precursor
charges 2–4. These are dataset-selection rules, separate from the model's
top-256 processing. Missing precursor charge is not inferred.

Manuscript FDR uses top-1 target–decoy competition with tied scores grouped at
one threshold. Use `shiftsearch-fdr` on the complete best-hit table for this
calculation. The convenience FDR output produced by search retains its
row-by-row implementation.

Peptide-sequence and peptidoform sets come from target PSMs accepted at the
specified PSM-level FDR; no separate peptide-level or phosphorylation-site
localization FDR is claimed. This release provides no E-value. Target/decoy
separation and synthetic smoke tests do not establish ground-truth peptide
correctness.
