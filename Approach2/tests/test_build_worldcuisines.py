"""Synthetic tests for build_worldcuisines.py: the zero-shot guard, the per-dish cap,
the stable prefix, the emitted stage-3 schema and the fixed-total mix (stdlib)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_worldcuisines as bw  # noqa: E402

STAGE3_FIELDS = {"id", "vg_image_id", "query", "answer",
                 "source_language", "nllb_lang_tag", "source_dataset"}


def raw(n, dishes=10):
    return [{"qa_id": i, "food_id": i % dishes, "open_ended_prompt": f"q{i}",
             "answer": f"a{i}", "image_url": f"http://x/{i}.jpg"} for i in range(n)]


class BuildWorldCuisinesTests(unittest.TestCase):
    def setUp(self):
        self._cached = bw.cache_image
        bw.cache_image = lambda url, name, images_dir, throttle=0.0: True
        self.addCleanup(lambda: setattr(bw, "cache_image", self._cached))

    def test_a_transfer_target_is_refused(self):
        for lang in ("jv_krama", "si_formal_spoken", "mn", "ga", "su_loma"):
            with self.assertRaisesRegex(SystemExit, "zero-shot transfer target"):
                bw.donor_tag(lang)

    def test_an_unmapped_donor_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "no NLLB tag"):
            bw.donor_tag("th")

    def test_donors_carry_their_nllb_tag(self):
        self.assertEqual(bw.donor_tag("bn"), ("Bengali", "ben_Beng"))

    def test_thumbnail_urls_collapse_to_the_original_file(self):
        thumb = ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/"
                 "Street.jpg/1279px-Street.jpg?download")
        self.assertEqual(bw.original_upload_url(thumb),
                         "https://upload.wikimedia.org/wikipedia/commons/a/ab/Street.jpg")

    def test_a_plain_upload_url_only_loses_its_query(self):
        plain = "https://upload.wikimedia.org/wikipedia/commons/2/23/Roll.jpg?download"
        self.assertEqual(bw.original_upload_url(plain),
                         "https://upload.wikimedia.org/wikipedia/commons/2/23/Roll.jpg")

    def test_questions_per_dish_are_capped(self):
        taken = bw.select(raw(200, dishes=10), sample=200, per_image=3, seed=13)
        per_dish = {}
        for row in taken:
            per_dish[row["food_id"]] = per_dish.get(row["food_id"], 0) + 1
        self.assertEqual(max(per_dish.values()), 3)
        self.assertEqual(len(taken), 30)

    def test_a_larger_sample_extends_the_same_prefix(self):
        small = bw.select(raw(400, dishes=100), sample=50, per_image=4, seed=13)
        large = bw.select(raw(400, dishes=100), sample=120, per_image=4, seed=13)
        self.assertEqual([r["qa_id"] for r in small], [r["qa_id"] for r in large[:50]])

    def test_emitted_rows_match_the_stage3_schema(self):
        rows = bw.build_rows(raw(5), "bn", Path("/tmp"))
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertEqual(set(row), STAGE3_FIELDS)
            self.assertEqual(row["source_language"], "Bengali")
            self.assertEqual(row["nllb_lang_tag"], "ben_Beng")
            self.assertTrue(row["vg_image_id"].startswith("wc_"))
        self.assertEqual(len({r["id"] for r in rows}), 5)

    def test_rows_without_a_question_or_answer_are_dropped(self):
        broken = raw(3)
        broken[0]["open_ended_prompt"] = "   "
        broken[1]["answer"] = None
        self.assertEqual(len(bw.build_rows(broken, "bn", Path("/tmp"))), 1)

    def test_the_mix_holds_the_total_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            gqa_path = Path(tmp) / "bn.jsonl"
            with gqa_path.open("w", encoding="utf-8") as handle:
                for i in range(100):
                    handle.write(json.dumps({"id": f"g{i}", "vg_image_id": str(i), "query": "q",
                                             "answer": "a", "source_language": "Bengali",
                                             "nllb_lang_tag": "ben_Beng",
                                             "source_dataset": "gqa"}) + "\n")
            cultural = bw.build_rows(raw(60, dishes=30), "bn", Path("/tmp"))
            mixed = bw.mix(cultural, gqa_path, 0.5, seed=13)
            self.assertEqual(len(mixed), 100)
            by_source = {}
            for row in mixed:
                by_source[row["source_dataset"]] = by_source.get(row["source_dataset"], 0) + 1
            self.assertEqual(by_source, {"gqa": 50, bw.SOURCE_DATASET: 50})

    def test_too_few_cultural_rows_aborts_instead_of_silently_shrinking(self):
        with tempfile.TemporaryDirectory() as tmp:
            gqa_path = Path(tmp) / "bn.jsonl"
            with gqa_path.open("w", encoding="utf-8") as handle:
                for i in range(100):
                    handle.write(json.dumps({"id": f"g{i}", "source_dataset": "gqa"}) + "\n")
            with self.assertRaisesRegex(SystemExit, "raise --sample"):
                bw.mix(bw.build_rows(raw(4, dishes=4), "bn", Path("/tmp")), gqa_path, 0.5, 13)


if __name__ == "__main__":
    unittest.main()
