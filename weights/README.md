# Pretrained inference weights

`shiftsearch.ckpt` contains the exact model tensors used for the paper's
searches. It is a compact inference export of the selected epoch-15
checkpoint, not a newly trained model. `metadata.json` records the source and
export hashes and confirms tensor equality.

The file omits optimizer, callback and logger state and machine-local paths.
Use it for inference with `configs/model.yaml`, not `--resume` training. Full
training-state checkpoints produced by your own runs can still be resumed.

Only load checkpoints you trust. The loader uses PyTorch's restricted
`weights_only=True` loading path. Search does not download any weights.
