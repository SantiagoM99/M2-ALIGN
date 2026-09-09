"""Exercise the real stage-3 loop on CPU with tiny local model/tokenizer doubles."""

import json
import logging
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Stage3LoopTests(unittest.TestCase):
    def test_actual_loop_resume_completion_and_best_selection(self):
        import torch
        from torch import nn
        import train_stage3_vqa as train
        from model import Mapping

        class Tok:
            pad_token = "pad"
            pad_token_id = 0
            eos_token = "eos"
            bos_token_id = 2
            chat_template = None

            def __call__(self, texts, **kw):
                single = isinstance(texts, str)
                texts = [texts] if single else texts
                rows = [[3 + ord(c) % 13 for c in t[:8]] for t in texts]
                n = max(map(len, rows))
                ids = [r + [0] * (n - len(r)) for r in rows]
                masks = [[1] * len(r) + [0] * (n - len(r)) for r in rows]
                if single:
                    return {"input_ids": rows[0], "attention_mask": [1] * len(rows[0])}
                return {
                    "input_ids": torch.tensor(ids),
                    "attention_mask": torch.tensor(masks),
                }

        class Toy(nn.Module):
            def __init__(self, *args, **kw):
                super().__init__()
                self.mapping_txt = Mapping(3, 4)
                self.mapping_vis = None
                self.model_mt = nn.Dropout(0.2)
                self.model_llm = nn.Identity()
                self.encoder_vis = None

            def forward(self, input_ids_mt, labels, **kw):
                x = input_ids_mt.float().mean(1, keepdim=True).expand(-1, 3) / 10
                v = (
                    self.mapping_txt(self.model_mt(x)).mean(1)
                    + self.mapping_txt.get_embed().mean()
                )
                return (v - labels.float().mean(1) / 10).square().mean()

        class Interrupted(Exception):
            pass

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / "data.jsonl"
            data.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": str(i),
                            "vg_image_id": str(i // 2),
                            "query": f"query{i}",
                            "answer": f"a{i % 3}",
                            "nllb_lang_tag": "eng_Latn",
                        }
                    )
                    + "\n"
                    for i in range(31)
                )
            )

            def args(out):
                return train.argument_parser().parse_args(
                    [
                        "--data-path",
                        str(data),
                        "--images-dir",
                        d,
                        "--output-dir",
                        str(out),
                        "--no-vision",
                        "--no-chat-template",
                        "--epochs",
                        "2",
                        "--train-batch-size",
                        "2",
                        "--grad-accum",
                        "2",
                        "--save-steps",
                        "2",
                        "--num-workers",
                        "0",
                        "--seed",
                        "13",
                    ]
                )

            original_save = train.save_snapshot

            def stop_after_first(*aa, **kk):
                original_save(*aa, **kk)
                raise Interrupted()

            with (
                patch.object(
                    train, "training_device", return_value=torch.device("cpu")
                ),
                patch.object(train, "DualEncoderMerger", Toy),
                patch.object(
                    train.NllbTokenizer, "from_pretrained", return_value=Tok()
                ),
                patch.object(
                    train.AutoTokenizer, "from_pretrained", return_value=Tok()
                ),
                patch.object(
                    train,
                    "git_state",
                    return_value={
                        "spec_sha": "test",
                        "code_sha": "test",
                        "dirty": False,
                    },
                ),
            ):
                train.main(args(root / "full"), logging.getLogger("test"))
                with patch.object(train, "save_snapshot", side_effect=stop_after_first):
                    with self.assertRaises(Interrupted):
                        train.main(args(root / "resumed"), logging.getLogger("test"))
                self.assertFalse((root / "resumed/complete.json").exists())
                train.main(args(root / "resumed"), logging.getLogger("test"))
                for filename in ("training_state.pt", "mapping/pytorch_model.bin"):
                    a = torch.load(root / "full" / filename, weights_only=False)
                    b = torch.load(root / "resumed" / filename, weights_only=False)
                    sa = (
                        a["mappings"]["mapping_txt"]
                        if filename == "training_state.pt"
                        else a["mapping_txt"]
                    )
                    sb = (
                        b["mappings"]["mapping_txt"]
                        if filename == "training_state.pt"
                        else b["mapping_txt"]
                    )
                    for k, v in sa.items():
                        self.assertTrue(torch.equal(v, sb[k]), k)
                self.assertEqual(a["loss"], b["loss"])
                check = args(root / "resumed")
                check.check_complete = True
                with self.assertRaises(SystemExit) as exit:
                    train.main(check, logging.getLogger("test"))
                self.assertEqual(exit.exception.code, 0)
                # A completed run does not initialize or retrain a model.
                with patch.object(
                    train,
                    "DualEncoderMerger",
                    side_effect=AssertionError("loaded completed model"),
                ):
                    train.main(args(root / "resumed"), logging.getLogger("test"))


if __name__ == "__main__":
    unittest.main()
