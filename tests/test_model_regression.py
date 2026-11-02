import json
from pathlib import Path
import unittest

import torch

from _model_fixture import fixture, snapshot


class ModelRegressionTests(unittest.TestCase):
    def test_matches_reference_golden(self):
        expected = json.loads((Path(__file__).parent / "fixtures" / "reference_cpu.json").read_text(encoding="utf-8"))
        actual = snapshot()
        self.assertEqual(actual["parameters"], 11825)
        self.assertEqual(actual["state_shapes"], expected["state_shapes"])
        for key in ("cell_mask", "cleavage_mask"):
            self.assertEqual(actual[key], expected[key], key)
        for key in ("b_cell_features", "y_cell_features", "b_selected_evidence", "y_selected_evidence", "logits", "scores"):
            torch.testing.assert_close(torch.tensor(actual[key]), torch.tensor(expected[key]),
                                       rtol=1e-6, atol=1e-25, msg=lambda msg: key + ": " + msg)

    def test_selected_score_is_proximity_weighted_and_padding_zero(self):
        model, raw = fixture()
        batch = model.prepare_batch(raw)
        for name in ("b_cell_features", "y_cell_features"):
            features = batch[name]
            values = model.encoder.cell_encoder(features)
            priority = (features[:, :1] * torch.sigmoid(model.encoder.peak_selector(values))).squeeze(1)
            expected = priority.masked_fill(~batch["cell_mask"], -torch.inf).max(-1).values
            expected = torch.where(batch["cleavage_mask"], expected, torch.zeros_like(expected))
            selected = model.encoder._select_peak(features, batch["cell_mask"], batch["cleavage_mask"])
            torch.testing.assert_close(selected[:, -1], expected)
            self.assertEqual(int(torch.count_nonzero(selected.masked_select(~batch["cleavage_mask"][:, None]))), 0)

    def test_backward_reaches_all_trainable_parameters(self):
        model, raw = fixture()
        model.train()
        batch = model.prepare_batch(raw)
        batch["label"] = torch.tensor([[1.0], [0.0]])
        loss = model._shared_step(batch)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)


if __name__ == "__main__":
    unittest.main()
