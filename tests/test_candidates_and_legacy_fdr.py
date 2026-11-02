from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ShiftSearch.search.database import enumerate_peptides_from_peptide_db, build_candidate_bucket_index, retrieve_candidates_by_mass_inmem
from ShiftSearch.search.fdr import compute_psm_q_values
from ShiftSearch.search.peptide_utils import build_prefix_masses, parse_mod_spec


class CandidateAndLegacyFDRTests(unittest.TestCase):
    def test_fixed_variable_sites_masses_and_decoy_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta = Path(directory) / "peptides.fasta"
            fasta.write_text(">target\nACDMPEPK\n>DECOY_reverse\nPEPMDCAK\n", encoding="utf-8")
            candidates = list(enumerate_peptides_from_peptide_db(
                fasta, parse_mod_spec("C:+57.021"), parse_mod_spec("M:+15.995"),
                min_len=6, max_len=32, max_variable_mods=1,
                decoy_prefix="DECOY_", max_pep_len=32))
            self.assertEqual([r[1] for r in candidates], ["AC+57.021DMPEPK", "AC+57.021DM+15.995PEPK",
                                                       "PEPMDC+57.021AK", "PEPM+15.995DC+57.021AK"])
            self.assertEqual([r[4] for r in candidates], [False, False, True, True])
            self.assertAlmostEqual(candidates[1][2] - candidates[0][2], 15.995, places=8)
            metas, buckets = build_candidate_bucket_index(candidates)
            found = retrieve_candidates_by_mass_inmem(metas, buckets, candidates[1][2], 10.0)
            self.assertEqual({r[1] for r in found}, {candidates[1][1], candidates[3][1]})
            sequence, prefix = build_prefix_masses(candidates[1][1], 32)
            self.assertEqual(sequence, "ACDMPEPK")
            self.assertEqual(prefix.shape, (33,))
            self.assertEqual(prefix.dtype, np.float32)

    def test_legacy_search_fdr_behavior_is_preserved(self):
        # This helper is preserved for old search outputs. Publication analysis
        # applies tied-score grouping separately; see test_paper_fdr.py.
        source = pd.DataFrame({"record_id": ["a", "b", "c", "d"],
                               "score": [0.9, 0.8, 0.7, 0.6],
                               "is_decoy": [False, True, False, False]})
        result = compute_psm_q_values(source)
        self.assertEqual(result["record_id"].tolist(), ["a", "b", "c", "d"])
        np.testing.assert_allclose(result["q_value"], [0.0, 1 / 3, 1 / 3, 1 / 3])
        empty = compute_psm_q_values(source.iloc[:0])
        self.assertEqual(len(empty), 0)
        self.assertIn("q_value", empty)


if __name__ == "__main__":
    unittest.main()
