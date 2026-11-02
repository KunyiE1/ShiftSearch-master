"""Publication FDR uses complete tied-score thresholds, not legacy row ranks."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ShiftSearch.evaluation.fdr import add_tie_aware_q_values
from ShiftSearch.cli.fdr import main as fdr_main


class PaperFDRTests(unittest.TestCase):
    def test_ties_share_threshold_and_q_value_and_preserve_ids(self):
        source = pd.DataFrame({"record_id": ["d1", "t1", "t2", "t3"],
                               "score": [0.9, 0.9, 0.8, 0.7],
                               "is_decoy": [True, False, False, False]})
        result = add_tie_aware_q_values(source)
        self.assertEqual(set(result["record_id"]), set(source["record_id"]))
        self.assertEqual(result.loc[result["score"].eq(0.9), "threshold_fdr"].nunique(), 1)
        self.assertEqual(result.loc[result["score"].eq(0.9), "q_value"].nunique(), 1)
        np.testing.assert_allclose(result["q_value"], [1 / 3] * 4)
        self.assertEqual(source.columns.tolist(), ["record_id", "score", "is_decoy"])

    def test_score_direction_and_empty_all_decoys(self):
        source = pd.DataFrame({"record_id": ["low", "medium", "high"],
                               "score": [0.01, 0.2, 0.5], "is_decoy": [False, False, True]})
        result = add_tie_aware_q_values(source, higher_is_better=False)
        self.assertEqual(result["record_id"].tolist(), ["low", "medium", "high"])
        np.testing.assert_allclose(result["q_value"], [0.0, 0.0, 0.5])
        result = add_tie_aware_q_values(source, higher_is_better=True)
        self.assertEqual(result["record_id"].tolist(), ["high", "medium", "low"])
        np.testing.assert_allclose(result["q_value"], [0.5, 0.5, 0.5])
        empty = add_tie_aware_q_values(source.iloc[:0])
        self.assertEqual(len(empty), 0)
        self.assertIn("q_value", empty)
        decoys = add_tie_aware_q_values(source.assign(is_decoy=True))
        self.assertTrue(decoys["q_value"].eq(1).all())
        targets = add_tie_aware_q_values(source.assign(is_decoy=False))
        self.assertTrue(targets["q_value"].eq(0).all())

    def test_nonfinite_scores_fail(self):
        for score in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                add_tie_aware_q_values(pd.DataFrame({"score": [score], "is_decoy": [False]}))

    def test_cli_parses_false_strings_and_writes_targets_only(self):
        with tempfile.TemporaryDirectory(prefix="shiftsearch-fdr-") as directory:
            base = Path(directory)
            source = pd.DataFrame({"spectrum_index": [0, 1, 2], "record_id": ["a", "b", "c"],
                                   "rank": [1, 1, 1], "score": [0.9, 0.8, 0.7],
                                   "is_decoy": ["False", "false", "True"]})
            source.to_csv(base / "best.tsv", sep="\t", index=False)
            fdr_main(["--input", str(base / "best.tsv"), "--output", str(base / "passing.tsv"),
                      "--all-output", str(base / "all.tsv"), "--fdr", "0.01"])
            passing = pd.read_csv(base / "passing.tsv", sep="\t")
            all_rows = pd.read_csv(base / "all.tsv", sep="\t")
            self.assertEqual(passing["record_id"].tolist(), ["a", "b"])
            self.assertEqual(set(all_rows["record_id"]), {"a", "b", "c"})
            self.assertTrue(passing["q_value"].eq(0).all())


if __name__ == "__main__":
    unittest.main()
