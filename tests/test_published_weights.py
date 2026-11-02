"""Keep the exact published inference artifact present and checkpoint-compatible."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import yaml
import ShiftSearch

from ShiftSearch.psm.hierarchical_by_module import PSMHierarchicalBY
from _model_fixture import snapshot


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "b4796308d5a90ea89d54343ff4c25b70179791bbf226c6702e49a47fc2f59dfb"
EXPORT_SHA256 = "d61801f351a4cd6cacd6dfc47305a78fb0eaadda31b498def41c92d066f85a40"


class PublishedWeightsTests(unittest.TestCase):
    def test_export_rejects_same_shape_but_different_proximity_scale(self):
        config = yaml.safe_load((REPOSITORY / "configs" / "model.yaml").read_text(encoding="utf-8"))
        config["Meta"]["fragment_tolerance_ppm"] = 40.0
        checkpoint = REPOSITORY / "weights" / "shiftsearch.ckpt"
        installed_parent = Path(ShiftSearch.__file__).resolve().parent.parent
        env = dict(os.environ, PYTHONPATH=str(installed_parent), CUDA_VISIBLE_DEVICES="",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1")
        with tempfile.TemporaryDirectory(prefix="shiftsearch-export-") as directory:
            temporary = Path(directory)
            supplied = temporary / "wrong_meta.yaml"
            supplied.write_text(yaml.safe_dump(config), encoding="utf-8")
            output = temporary / "new_export.ckpt"
            metadata = temporary / "new_metadata.json"
            completed = subprocess.run(
                [sys.executable, str(REPOSITORY / "scripts" / "export_inference_checkpoint.py"),
                 "--input", str(checkpoint), "--config", str(supplied),
                 "--output", str(output), "--metadata", str(metadata)],
                cwd=temporary, env=env, capture_output=True, text=True, timeout=60,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("meta_params differs", completed.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(metadata.exists())
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), EXPORT_SHA256)

    def test_metadata_and_export_integrity(self):
        metadata = json.loads((REPOSITORY / "weights" / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["model"], "ShiftSearch")
        self.assertEqual(metadata["export_file"], "shiftsearch.ckpt")
        checkpoint = REPOSITORY / "weights" / metadata["export_file"]
        self.assertTrue(checkpoint.is_file(), "The published inference weights must be shipped in the repository")
        self.assertEqual(metadata["source_checkpoint_sha256"], SOURCE_SHA256)
        self.assertEqual(metadata["source_checkpoint_name"], "best-epoch=15-step=46880-val_acc=0.970270.ckpt")
        self.assertEqual(metadata["export_sha256"], EXPORT_SHA256)
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), EXPORT_SHA256)
        self.assertEqual(checkpoint.stat().st_size, metadata["export_bytes"])
        self.assertEqual(metadata["export_bytes"], 51568)
        self.assertEqual(metadata["parameters"], 11825)
        self.assertEqual(metadata["scope"], "inference_only")
        self.assertIs(metadata["all_source_model_tensors_identical"], True)

    def test_safe_strict_load_and_original_paper_logits(self):
        checkpoint_path = REPOSITORY / "weights" / "shiftsearch.ckpt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.assertFalse({"optimizer_states", "lr_schedulers", "callbacks", "loops"}.intersection(checkpoint))
        config = yaml.safe_load((REPOSITORY / "configs" / "model.yaml").read_text(encoding="utf-8"))
        model = PSMHierarchicalBY(config["Model"]["params"], config["Train"], config["Meta"]).cpu().eval()
        incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(len(checkpoint["state_dict"]), 9)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11825)
        for tensor in checkpoint["state_dict"].values():
            self.assertTrue(bool(torch.isfinite(tensor).all()))
        # Original paper checkpoint run on the documented small CPU fixture.
        # No original bundle is required; tolerance allows portable CPU kernels.
        result = snapshot(checkpoint_path)
        torch.testing.assert_close(torch.tensor(result["logits"]),
                                   torch.tensor([6.305037021636963, 6.415328502655029]),
                                   rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(torch.tensor(result["scores"]),
                                   torch.tensor([0.9981762170791626, 0.9983664155006409]),
                                   rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
