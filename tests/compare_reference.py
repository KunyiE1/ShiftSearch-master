"""Optional maintainer audit against an explicitly supplied original bundle.

The normal test suite is self-contained. This additional audit loads the same
checkpoint in the original bundle and the installed package in separate Python
processes, comparing state tensors, cell features, selection and final scores.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bundle", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--installed-checkpoint", type=Path, help="Optional compact export to compare with the original checkpoint.")
    args = parser.parse_args()
    tests = Path(__file__).resolve().parent
    checkpoint = str(args.checkpoint.resolve()) if args.checkpoint else None
    installed_checkpoint = str(args.installed_checkpoint.resolve()) if args.installed_checkpoint else checkpoint
    program = (
        "import sys,json;"
        "reference=json.loads(sys.argv[1]);"
        "sys.path.insert(0,reference) if reference else None;"
        "sys.path.insert(1,sys.argv[2]);"
        "from _model_fixture import snapshot;"
        "import ShiftSearch;"
        "value=snapshot(json.loads(sys.argv[3]));"
        "print(json.dumps(dict(origin=ShiftSearch.__file__,snapshot=value)))"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1")
    with tempfile.TemporaryDirectory(prefix="shiftsearch-parity-") as temporary:
        outputs = []
        for reference, state_path in ((str(args.reference_bundle.resolve()), checkpoint), (None, installed_checkpoint)):
            completed = subprocess.run([sys.executable, "-c", program, json.dumps(reference), str(tests), json.dumps(state_path)],
                                       env=env, cwd=temporary, capture_output=True, text=True, timeout=180)
            if completed.returncode:
                raise RuntimeError(completed.stdout + "\n" + completed.stderr)
            outputs.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    if outputs[0]["snapshot"] != outputs[1]["snapshot"]:
        raise AssertionError("Original and installed model snapshots differ")
    print(json.dumps({"status": "PASS", "original_import": outputs[0]["origin"],
                      "installed_import": outputs[1]["origin"],
                      "checkpoint": checkpoint,
                      "installed_checkpoint": installed_checkpoint,
                      "parameters": outputs[1]["snapshot"]["parameters"],
                      "state_tensor_sha256": outputs[1]["snapshot"]["state_tensor_sha256"],
                      "logits": outputs[1]["snapshot"]["logits"],
                      "scores": outputs[1]["snapshot"]["scores"]}, indent=2))


if __name__ == "__main__":
    main()
