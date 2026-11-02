"""Check the shipped real-spectrum example without treating it as a benchmark."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import ShiftSearch


REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples" / "celegans"
SPEC = importlib.util.spec_from_file_location("check_search_example", REPOSITORY / "scripts" / "check_search_example.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def load_manifest():
    return json.loads((EXAMPLE / "manifest.json").read_text(encoding="utf-8"))


def unique_input(manifest, suffixes):
    paths = [EXAMPLE / name for name in manifest["files"] if Path(name).suffix.lower() in suffixes]
    if len(paths) != 1:
        raise AssertionError(f"Expected one input file with suffix in {suffixes}, found {paths}")
    return paths[0]


class RealExampleTests(unittest.TestCase):
    def test_manifest_input_integrity_and_expected_records(self):
        manifest = load_manifest()
        self.assertTrue({"spectra.mgf", "peptides.fasta", "search.yaml", "expected_best.tsv"}.issubset(manifest["files"]))
        for name, information in manifest["files"].items():
            path = (EXAMPLE / name).resolve()
            self.assertTrue(path.is_relative_to(EXAMPLE.resolve()), name)
            content = path.read_bytes()
            self.assertEqual(len(content), information["bytes"], name)
            self.assertEqual(hashlib.sha256(content).hexdigest(), information["sha256"], name)
        mgf = unique_input(manifest, {".mgf"})
        self.assertEqual(sum(line.strip() == "BEGIN IONS" for line in mgf.read_text(encoding="utf-8").splitlines()), manifest["n_spectra"])
        expected = CHECKER.read_results(EXAMPLE / "expected_best.tsv")
        self.assertEqual(len(expected), manifest["expected_best_hits"])
        self.assertEqual(len(manifest["spectra"]), manifest["n_spectra"])
        spectrum_keys = {(item["title"], int(item["charge"])) for item in manifest["spectra"]}
        self.assertEqual(len(spectrum_keys), manifest["n_spectra"])
        searchable = {(item["title"], int(item["charge"])) for item in manifest["spectra"] if item["candidate_count"] > 0}
        self.assertEqual(set(expected), searchable)

    def test_cpu_search_matches_recorded_best_hits(self):
        manifest = load_manifest()
        installed_parent = Path(ShiftSearch.__file__).resolve().parent.parent
        env = dict(os.environ, PYTHONPATH=str(installed_parent), CUDA_VISIBLE_DEVICES="",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1")
        with tempfile.TemporaryDirectory(prefix="shiftsearch-real-example-") as directory:
            temporary = Path(directory)
            output = temporary / "results.tsv"
            arguments = [sys.executable, "-m", "ShiftSearch.cli.search",
                         "--mgf", str(unique_input(manifest, {".mgf"})),
                         "--peptide-fasta", str(unique_input(manifest, {".fasta", ".fa"})),
                         "--checkpoint", str(REPOSITORY / "weights" / "shiftsearch.ckpt"),
                         "--config", str(REPOSITORY / "configs" / "model.yaml"),
                         "--search-config", str(EXAMPLE / "search.yaml"),
                         "--output", str(output), "--device", "cpu", "--batch-size", "32",
                         "--spectra-per-batch", "8", "--fdr", "1"]
            completed = subprocess.run(arguments, cwd=temporary, env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(completed.returncode, 0, completed.stdout + "\n" + completed.stderr)
            report = CHECKER.check_results(output.with_name("results.best.tsv"), EXAMPLE / "expected_best.tsv", 0.0005)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["best_hits"], manifest["expected_best_hits"])


if __name__ == "__main__":
    unittest.main()
