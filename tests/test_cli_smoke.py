"""Exercise installed CLIs on three synthetic spectra, never research datasets."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import ShiftSearch


class InstalledCLISmokeTests(unittest.TestCase):
    def test_cpu_train_checkpoint_and_search_from_external_directory(self):
        repository = Path(__file__).resolve().parents[1]
        example = repository / "examples" / "create_tiny_demo.py"
        installed_parent = Path(ShiftSearch.__file__).resolve().parent.parent
        env = dict(os.environ)
        # Preserve only the explicitly imported installation for child imports;
        # do not let a parent research checkout supply undeclared modules.
        env["PYTHONPATH"] = str(installed_parent)
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1")
        with tempfile.TemporaryDirectory(prefix="shiftsearch-smoke-") as temporary:
            work = Path(temporary)
            fixture = work / "synthetic"

            def run(arguments):
                completed = subprocess.run([sys.executable, *map(str, arguments)], cwd=work,
                                           env=env, capture_output=True, text=True, timeout=180)
                self.assertEqual(completed.returncode, 0, completed.stdout + "\n" + completed.stderr)
                return completed

            run([example, "--output-dir", fixture])
            notice = json.loads((fixture / "SYNTHETIC_FIXTURE.json").read_text(encoding="utf-8"))
            self.assertEqual(notice["spectra"], 3)
            self.assertEqual(notice["parameters"], 11825)
            run(["-m", "ShiftSearch.cli.train", "--data", fixture / "spectra.mgf",
                 "--train-pairs", fixture / "train_pairs.tsv", "--val-pairs", fixture / "val_pairs.tsv",
                 "--config", fixture / "model.yaml", "--devices", "1", "--n-workers", "0", "--n-epochs", "1"])
            checkpoint = fixture / "checkpoints" / "last.ckpt"
            self.assertTrue(checkpoint.is_file())
            # Both files were created in this test, not downloaded/untrusted.
            trained = torch.load(checkpoint, map_location="cpu", weights_only=False)
            initial = torch.load(fixture / "synthetic_untrained.ckpt", map_location="cpu", weights_only=False)
            self.assertEqual(trained["global_step"], 2)
            self.assertEqual(set(trained["state_dict"]), set(initial["state_dict"]))
            self.assertTrue(any(not torch.equal(trained["state_dict"][key], value)
                                for key, value in initial["state_dict"].items()))
            run(["-m", "ShiftSearch.cli.search", "--mgf", fixture / "spectra.mgf",
                 "--peptide-fasta", fixture / "peptides.fasta", "--checkpoint", checkpoint,
                 "--config", fixture / "model.yaml", "--search-config", fixture / "search.yaml",
                 "--output", fixture / "results.tsv", "--device", "cpu"])
            with (fixture / "results.tsv").open(encoding="utf-8", newline="") as handle:
                candidates = list(csv.DictReader(handle, delimiter="\t"))
            with (fixture / "results.best.tsv").open(encoding="utf-8", newline="") as handle:
                best = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(candidates), 6)
            self.assertEqual(len(best), 3)
            self.assertEqual({int(row["spectrum_index"]) for row in best}, {0, 1, 2})
            self.assertTrue(all(0.0 <= float(row["score"]) <= 1.0 for row in best))
            self.assertTrue(all(int(row["rank"]) == 1 for row in best))


if __name__ == "__main__":
    unittest.main()
