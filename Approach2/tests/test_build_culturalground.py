"""Synthetic tests for build_culturalground.py: zero-shot guard, language filter,
per-image cap, safe tar extraction, emitted schema and the fixed-total mix (stdlib)."""

import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_culturalground as cg  # noqa: E402

STAGE3_FIELDS = {"id", "vg_image_id", "query", "answer",
                 "source_language", "nllb_lang_tag", "source_dataset"}


def raw(n, lang="bn", images=10):
    return [{"id": f"q{i}", "language": lang, "image": f"img{i % images}.jpg",
             "question": f"what is this {i}?", "answer": f"a{i}"} for i in range(n)]


def encoded(name):
    """A real, decodable image, because the builder now decodes what it extracts."""
    from PIL import Image

    suffix = Path(name).suffix.lower()
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(
        buffer, format={".png": "PNG", ".tiff": "TIFF", ".gif": "GIF"}.get(suffix, "JPEG")
    )
    return buffer.getvalue()


def archive(tmp, names, evil=False, broken=()):
    """A country archive; with evil=True it also carries a path-traversal member.

    Names in `broken` carry bytes that do not decode, which is how
    CulturalGround's truncated JPEGs behave.
    """
    path = Path(tmp) / "country.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name in names:
            data = b"truncated" if name in broken else encoded(name)
            info = tarfile.TarInfo(f"images/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if evil:
            data = b"pwned"
            info = tarfile.TarInfo("../../escaped.jpg")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


class CulturalGroundTests(unittest.TestCase):
    def test_a_transfer_target_is_refused(self):
        for lang in ("jv", "si", "mn", "ga", "su"):
            with self.assertRaisesRegex(SystemExit, "zero-shot transfer target"):
                cg.donor_tag(lang)

    def test_donors_carry_their_nllb_tag(self):
        self.assertEqual(cg.donor_tag("bn"), ("Bengali", "ben_Beng"))

    def test_rows_in_other_languages_are_filtered_out(self):
        rows = raw(20, lang="bn") + raw(20, lang="en")
        taken = cg.select(rows, "bn", sample=100, per_image=10, seed=13)
        self.assertEqual(len(taken), 20)
        self.assertTrue(all(r["language"] == "bn" for r in taken))

    def test_a_missing_language_names_what_is_there(self):
        with self.assertRaisesRegex(SystemExit, r"no rows matching \['ko', 'korean'\]"):
            cg.select(raw(5, lang="bn"), "ko", sample=5, per_image=4, seed=13)

    def test_the_language_may_be_spelled_as_a_name(self):
        """CulturalGround may write `Bengali` where we pass `bn`; both must match,
        and a language whose name merely starts the same must not."""
        taken = cg.select(raw(20, lang="Bengali") + raw(20, lang="en"), "bn",
                          sample=100, per_image=10, seed=13)
        self.assertEqual(len(taken), 20)
        with self.assertRaises(SystemExit):
            cg.select(raw(5, lang="Bulgarian"), "bn", sample=5, per_image=4, seed=13)

    def test_the_literal_string_None_counts_as_missing(self):
        """CulturalGround writes "None" where there is no value, and a bare str()
        would put that word into a prompt or a filename."""
        self.assertIsNone(cg.pick({"image": "None"}, "image"))
        self.assertEqual(cg.pick({"image": "india/Q1_x.jpg"}, "image"), "india/Q1_x.jpg")
        self.assertIsNone(cg.pick({"original_question": "None"}, "question"))

    def test_media_is_not_treated_as_a_filename(self):
        """`media` is the Commons page name ("Narendra_Modi"), with no directory
        and no extension; it would never match an archive member."""
        self.assertIsNone(cg.pick({"image": "None", "media": "Narendra_Modi"}, "image"))

    def test_the_english_answer_is_joined_by_entity_property_and_type(self):
        """Our stage-3 rows are native question with English answer, and
        CulturalGround's label is in the row's own language."""
        rows = [
            {"id": "Q1", "property_id": "None", "question_type": "entity_level_vqa",
             "language": "en", "label": "Narendra Modi", "reformulated_question": "Who?",
             "image": "india/m1.jpg"},
            {"id": "Q1", "property_id": "P6", "question_type": "property_level_vqa",
             "language": "en", "label": "India", "reformulated_question": "Where?",
             "image": "india/m1.jpg"},
            {"id": "Q1", "property_id": "None", "question_type": "entity_level_vqa",
             "language": "bn", "label": "নরেন্দ্র মোদী", "reformulated_question": "কে?",
             "image": "india/m1.jpg"},
        ]
        answers = cg.english_answers(rows)
        self.assertEqual(answers[("Q1", "None", "entity_level_vqa")], "Narendra Modi")
        self.assertEqual(answers[("Q1", "P6", "property_level_vqa")], "India")
        bn = [r for r in rows if r["language"] == "bn"]
        with tempfile.TemporaryDirectory() as tmp:
            cg.download = lambda template, country: archive(tmp, ["m1.jpg"])
            images = Path(tmp) / "images"
            images.mkdir()
            built = cg.build_rows(bn, "bn", "india", images, answers)
        self.assertEqual([(r["query"], r["answer"]) for r in built], [("কে?", "Narendra Modi")])

    def test_an_entity_with_two_conflicting_english_answers_is_dropped(self):
        """Ambiguity is skipped rather than resolved by picking one."""
        rows = [
            {"id": "Q1", "property_id": "None", "question_type": "t", "language": "en",
             "label": "A", "image": "india/m1.jpg"},
            {"id": "Q1", "property_id": "None", "question_type": "t", "language": "en",
             "label": "B", "image": "india/m1.jpg"},
        ]
        self.assertEqual(cg.english_answers(rows), {})

    def test_a_format_the_trainer_cannot_open_is_dropped(self):
        """train_stage3_vqa appends .jpg/.jpeg/.png to the id, so a .tiff row
        trains for fifteen minutes and dies in a DataLoader worker (job 22750351)."""
        self.assertIsNone(cg.loadable_name("Q1_airport.tiff"))
        self.assertIsNone(cg.loadable_name("Q1_map.svg"))
        self.assertIsNone(cg.loadable_name("Q1_noextension"))
        with tempfile.TemporaryDirectory() as tmp:
            cg.download = lambda template, country: archive(tmp, ["ok.jpg", "bad.tiff"])
            images = Path(tmp) / "images"
            images.mkdir()
            rows = [{"id": "q0", "language": "bn", "image": "india/ok.jpg",
                     "question": "what?", "answer": "a"},
                    {"id": "q1", "language": "bn", "image": "india/bad.tiff",
                     "question": "what?", "answer": "a"}]
            built = cg.build_rows(rows, "bn", "country", images)
            self.assertEqual([r["vg_image_id"] for r in built], ["cg_ok"])
            self.assertFalse((images / "cg_bad.tiff").exists())

    def test_a_case_variant_extension_is_kept_under_a_lowercased_name(self):
        """.JPG is the same bytes under a name the trainer will not try."""
        self.assertEqual(cg.loadable_name("Q1_photo.JPG"), "Q1_photo.jpg")
        with tempfile.TemporaryDirectory() as tmp:
            cg.download = lambda template, country: archive(tmp, ["Shot.JPG"])
            images = Path(tmp) / "images"
            images.mkdir()
            rows = [{"id": "q0", "language": "bn", "image": "india/Shot.JPG",
                     "question": "what?", "answer": "a"}]
            built = cg.build_rows(rows, "bn", "country", images)
            self.assertEqual([r["vg_image_id"] for r in built], ["cg_Shot"])
            self.assertTrue((images / "cg_Shot.jpg").exists())

    def test_an_image_that_does_not_decode_is_dropped_and_deleted(self):
        """CulturalGround ships truncated JPEGs: one is exactly 5 MiB and stops
        mid-scan, which cost three hours of training (job 22777617). The
        extension is not evidence that the bytes decode."""
        with tempfile.TemporaryDirectory() as tmp:
            cg.download = lambda template, country: archive(
                tmp, ["broken.jpg"], broken={"broken.jpg"})
            images = Path(tmp) / "images"
            images.mkdir()
            rows = [{"id": "q0", "language": "bn", "image": "india/broken.jpg",
                     "question": "what?", "answer": "a"}]
            built = cg.build_rows(rows, "bn", "country", images)
            self.assertEqual(built, [])
            self.assertFalse((images / "cg_broken.jpg").exists())

    def test_opens_distinguishes_a_real_image_from_bytes_with_the_right_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.png"
            from PIL import Image
            Image.new("RGB", (4, 4), (1, 2, 3)).save(good)
            bad = Path(tmp) / "bad.png"
            bad.write_bytes(b"not an image")
            self.assertTrue(cg.opens(good))
            self.assertFalse(cg.opens(bad))

    def test_questions_per_image_are_capped(self):
        taken = cg.select(raw(120, images=10), "bn", sample=120, per_image=3, seed=13)
        self.assertEqual(len(taken), 30)

    def test_a_larger_sample_extends_the_same_prefix(self):
        small = cg.select(raw(400, images=100), "bn", sample=40, per_image=4, seed=13)
        large = cg.select(raw(400, images=100), "bn", sample=90, per_image=4, seed=13)
        self.assertEqual([r["id"] for r in small], [r["id"] for r in large[:40]])

    def test_extraction_takes_only_wanted_members_and_never_escapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = archive(tmp, ["img0.jpg", "img1.jpg", "unwanted.jpg"], evil=True)
            cg.download = lambda template, country: arc
            images = Path(tmp) / "images"
            images.mkdir()
            written = cg.extract_images("country", {"img0.jpg", "img1.jpg"}, images)
            self.assertEqual(written, {"img0.jpg", "img1.jpg"})
            self.assertEqual(sorted(p.name for p in images.iterdir()),
                             ["cg_img0.jpg", "cg_img1.jpg"])
            self.assertFalse((Path(tmp).parent / "escaped.jpg").exists())

    def test_emitted_rows_match_the_stage3_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = archive(tmp, [f"img{i}.jpg" for i in range(10)])
            cg.download = lambda template, country: arc
            images = Path(tmp) / "images"
            images.mkdir()
            rows = cg.build_rows(raw(10), "bn", "country", images)
            self.assertEqual(len(rows), 10)
            for row in rows:
                self.assertEqual(set(row), STAGE3_FIELDS)
                self.assertEqual(row["nllb_lang_tag"], "ben_Beng")
                self.assertTrue(row["vg_image_id"].startswith("cg_"))

    def test_rows_whose_image_is_absent_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc = archive(tmp, ["img0.jpg"])
            cg.download = lambda template, country: arc
            images = Path(tmp) / "images"
            images.mkdir()
            rows = cg.build_rows(raw(4, images=4), "bn", "country", images)
            self.assertEqual([r["vg_image_id"] for r in rows], ["cg_img0"])

    def test_the_mix_holds_the_total_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            gqa = Path(tmp) / "bn.jsonl"
            with gqa.open("w", encoding="utf-8") as handle:
                for i in range(100):
                    handle.write(json.dumps({"id": f"g{i}", "source_dataset": "gqa"}) + "\n")
            cultural = [{"id": f"c{i}", "source_dataset": cg.SOURCE_DATASET} for i in range(60)]
            mixed = cg.mix(cultural, gqa, 0.5, seed=13)
            self.assertEqual(len(mixed), 100)
            self.assertEqual(sum(1 for r in mixed if r["source_dataset"] == cg.SOURCE_DATASET), 50)

    def test_too_few_cultural_rows_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            gqa = Path(tmp) / "bn.jsonl"
            with gqa.open("w", encoding="utf-8") as handle:
                for i in range(100):
                    handle.write(json.dumps({"id": f"g{i}", "source_dataset": "gqa"}) + "\n")
            with self.assertRaisesRegex(SystemExit, "raise --sample"):
                cg.mix([{"id": "c", "source_dataset": cg.SOURCE_DATASET}], gqa, 0.5, 13)


if __name__ == "__main__":
    unittest.main()
