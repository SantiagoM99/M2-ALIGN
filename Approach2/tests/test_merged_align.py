"""CPU tests for the vision-alignment trainer's data path, on a stub processor.

The loop itself needs a GPU, but everything that decides whether the loop trains
the right thing is pure: how many placeholder positions the prompt reserves, that
the loss is scored on the caption and nothing else, that padding never turns into
a scored token, and that the split is reproducible.

    python Approach2/tests/test_merged_align.py
"""

import sys
import unittest
from pathlib import Path

import torch

MERGED = Path(__file__).resolve().parents[1] / "merged"
sys.path.insert(0, str(MERGED))
from align_vision import IGNORE, build_example, pad_batch, split_rows  # noqa: E402

PLACEHOLDER = "<|vision_pad|>"
PLACEHOLDER_ID = 901
IMAGE_ID = 900
PAD_ID = 0


class StubTokenizer:
    pad_token_id = PAD_ID
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False, return_tensors=None,
                 truncation=False, max_length=None):
        ids = [ord(c) % 500 + 10 for c in text]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids])}


class StubProcessor:
    """Expands the reserved placeholders into ids the way a real tokenizer would."""

    tokenizer = StubTokenizer()

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            return_dict=True, return_tensors=None):
        text = messages[0]["content"][1]["text"]
        reserved = text.count(PLACEHOLDER)
        ids = [1] + [IMAGE_ID] * 4 + [PLACEHOLDER_ID] * reserved + [3, 4]
        return {
            "input_ids": torch.tensor([ids]),
            "pixel_values": torch.zeros(1, 3, 8, 8),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }


class AlignDataTests(unittest.TestCase):
    def setUp(self):
        self.processor = StubProcessor()

    def build(self, caption="a dish", n=5):
        return build_example(self.processor, caption, object(), n, PLACEHOLDER, 64)

    def test_the_prompt_reserves_exactly_the_requested_placeholders(self):
        example = self.build(n=5)
        self.assertEqual(int((example["input_ids"] == PLACEHOLDER_ID).sum()), 5)

    def test_no_placeholders_are_reserved_for_early_mode(self):
        example = self.build(n=0)
        self.assertEqual(int((example["input_ids"] == PLACEHOLDER_ID).sum()), 0)

    def test_only_the_caption_is_scored(self):
        example = self.build(caption="abc", n=3)
        labels, ids = example["labels"], example["input_ids"]
        scored = labels != IGNORE
        self.assertEqual(int(scored.sum()), 3)
        self.assertTrue(torch.equal(labels[scored], ids[scored]))
        self.assertTrue(bool(scored[-3:].all()))
        self.assertFalse(bool(scored[:-3].any()))

    def test_a_long_caption_is_truncated_not_dropped(self):
        example = build_example(self.processor, "x" * 100, object(), 0, PLACEHOLDER, 12)
        self.assertEqual(int((example["labels"] != IGNORE).sum()), 12)

    def test_the_mask_covers_every_real_token(self):
        example = self.build()
        self.assertEqual(int(example["attention_mask"].sum()), example["input_ids"].shape[0])

    def test_padding_never_becomes_a_scored_token(self):
        batch = pad_batch([self.build(caption="ab"), self.build(caption="abcdef")], PAD_ID)
        width = batch["input_ids"].shape[1]
        self.assertEqual(batch["labels"].shape, (2, width))
        padded = batch["attention_mask"] == 0
        self.assertTrue(bool((batch["labels"][padded] == IGNORE).all()))
        self.assertTrue(bool((batch["input_ids"][padded] == PAD_ID).all()))

    def test_padding_keeps_one_image_grid_per_example(self):
        batch = pad_batch([self.build(), self.build()], PAD_ID)
        self.assertEqual(batch["image_grid_thw"].shape, (2, 3))
        self.assertEqual(batch["pixel_values"].shape[0], 2)

    def test_the_split_is_reproducible_and_the_right_size(self):
        rows = [{"image_url": f"u{i}", "target_caption": "c"} for i in range(100)]
        train, val = split_rows(rows, 0.03, 13)
        again = split_rows(list(reversed(rows)), 0.03, 13)
        self.assertEqual(len(val), 3)
        self.assertEqual(len(train), 97)
        self.assertEqual([r["image_url"] for r in val], [r["image_url"] for r in again[1]])
        self.assertFalse(set(r["image_url"] for r in train) & set(r["image_url"] for r in val))

    def test_a_tiny_corpus_still_leaves_a_validation_example(self):
        rows = [{"image_url": f"u{i}", "target_caption": "c"} for i in range(4)]
        train, val = split_rows(rows, 0.03, 13)
        self.assertEqual(len(val), 1)
        self.assertEqual(len(train), 3)


if __name__ == "__main__":
    unittest.main()
