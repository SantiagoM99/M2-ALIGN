"""The overlap audit must see exposure where it exists and not where it does not (stdlib)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "analysis"))
import cg_entity_overlap as audit  # noqa: E402

CULTURAL = audit.CULTURAL


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.training = write(
            self.root / "bn_cg50.jsonl",
            [
                {"vg_image_id": "cg_Q42_Ganesh_statue", "answer": "Ganesh",
                 "source_dataset": CULTURAL},
                {"vg_image_id": "cg_Q43_Saraswati_temple", "answer": "Saraswati",
                 "source_dataset": CULTURAL},
                {"vg_image_id": "2375767", "answer": "bread",
                 "source_dataset": "gqa_balanced_train_translated"},
            ],
        )
        self.exposure = audit.training_exposure(self.training)

    def test_the_cultural_half_is_what_counts(self):
        self.assertEqual(self.exposure["rows_total"], 3)
        self.assertEqual(self.exposure["rows_cultural"], 2)
        self.assertNotIn("bread", self.exposure["answers"])

    def test_a_file_without_cultural_rows_is_refused(self):
        plain = write(self.root / "bn.jsonl", [{"answer": "x", "source_dataset": "gqa"}])
        with self.assertRaises(SystemExit):
            audit.training_exposure(plain)

    def test_an_overlapping_gold_answer_is_counted(self):
        rows = [
            {"choices": ["Ganesh", "Kartik"], "answer_index": 0},
            {"choices": ["Kartik", "Ganesh"], "answer_index": 0},
        ]
        out = audit.audit_panel(rows, self.exposure)
        self.assertEqual(out["gold_answer_in_training_answers"], 1)
        self.assertEqual(out["gold_answer_share"], 0.5)
        # The second item's gold is absent but a distractor is present.
        self.assertEqual(out["any_choice_in_training_answers"], 2)

    def test_the_entity_label_is_read_from_the_image_id(self):
        """The Commons stem is the only entity label the training rows carry."""
        out = audit.audit_panel([{"choices": ["statue"], "answer_index": 0}], self.exposure)
        self.assertEqual(out["gold_answer_in_training_entity_labels"], 1)
        self.assertEqual(out["gold_answer_in_training_answers"], 0)

    def test_case_accents_and_punctuation_do_not_hide_an_overlap(self):
        out = audit.audit_panel([{"choices": ["  gánesh! "], "answer_index": 0}], self.exposure)
        self.assertEqual(out["gold_answer_in_training_answers"], 1)

    def test_a_disjoint_panel_scores_zero(self):
        out = audit.audit_panel([{"choices": ["borobudur", "prambanan"], "answer_index": 0}],
                                self.exposure)
        self.assertEqual(out["gold_answer_share"], 0.0)
        self.assertEqual(out["any_choice_share"], 0.0)

    def test_a_panel_without_choices_is_refused(self):
        with self.assertRaises(SystemExit):
            audit.audit_panel([{"answer_index": 0}], self.exposure)


if __name__ == "__main__":
    unittest.main()
