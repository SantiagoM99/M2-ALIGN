import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1_contract import shuffle_maps, validate_shuffle


class ShuffleTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "id": f"{s}-{i}-{q}",
                "image_id": f"{s}-{i}",
                "subset": s,
                "query": "native",
            }
            for s in ["a", "b"]
            for i in range(5)
            for q in range(2)
        ]

    def test_pairing_order_and_distinct_per_subset(self):
        rows = self.rows()
        maps = shuffle_maps(rows, "cvqa")
        self.assertEqual(maps, shuffle_maps(list(reversed(rows)), "cvqa"))
        for m in maps:
            for r in rows:
                x = m["items"][r["id"]]
                self.assertNotEqual(x["original_image_id"], x["assigned_image_id"])
                self.assertTrue(x["assigned_image_id"].startswith(r["subset"]))
                sibling = r["id"][:-1] + "1"
                self.assertEqual(x, m["items"][sibling])
        for s in ["a", "b"]:
            self.assertEqual(
                3,
                len(
                    {
                        tuple(
                            m["items"][r["id"]]["assigned_image_id"]
                            for r in rows
                            if r["subset"] == s
                        )
                        for m in maps
                    }
                ),
            )

    def test_translations_share_maps_and_tampering_aborts(self):
        rows = self.rows()
        translated = copy.deepcopy(rows)
        for r in translated:
            r["query"] = "translated"
            r["nllb_lang_tag"] = "eng_Latn"
        maps = shuffle_maps(rows, "xgqa")
        self.assertEqual(maps, shuffle_maps(translated, "xgqa"))
        maps[0]["items"][rows[0]["id"]]["assigned_image_id"] = "wrong"
        with self.assertRaises(ValueError):
            validate_shuffle(maps[0], rows, "xgqa")
        with self.assertRaises(ValueError):
            shuffle_maps(rows + rows[:1], "cvqa")
        with self.assertRaises(ValueError):
            shuffle_maps(rows[:4], "cvqa")


class ModelTests(unittest.TestCase):
    def test_strict_branch_loading_and_legacy_gate(self):
        import torch
        from model import Mapping
        from common import load_branch_checkpoint

        with tempfile.TemporaryDirectory() as d:
            a, b = Mapping(3, 4), Mapping(3, 4)
            with torch.no_grad():
                a.gate.fill_(7)
            state = {k: v.clone() for k, v in b.state_dict().items() if k != "gate"}
            path = Path(d) / "old.pt"
            torch.save({"mapping_txt": state, "mapping_vis": a.state_dict()}, path)
            load_branch_checkpoint(path, a, "mapping_txt")
            self.assertEqual(a.gate.item(), 1)
            self.assertTrue(torch.equal(a.end_boundary, b.end_boundary))
            del state["mlp.linear1.weight"]
            torch.save({"mapping_txt": state}, path)
            with self.assertRaises(RuntimeError):
                load_branch_checkpoint(path, a, "mapping_txt")

    def test_prompt_only_forward_generate_and_encoder_absence(self):
        import torch
        from unittest.mock import patch
        from transformers import Gemma2Config, Gemma2ForCausalLM
        from model import DualEncoderMerger

        config = Gemma2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
        )
        llm = Gemma2ForCausalLM(config).to(torch.bfloat16).eval()
        with (
            patch("model.AutoModelForCausalLM.from_pretrained", return_value=llm),
            patch(
                "model.M2M100Model.from_pretrained",
                side_effect=AssertionError("NLLB loaded"),
            ),
            patch(
                "model.AutoModel.from_pretrained",
                side_effect=AssertionError("vision loaded"),
            ),
        ):
            m = DualEncoderMerger(
                None,
                None,
                "tiny",
                8,
                2,
                0,
                use_text_branch=False,
                use_vision_branch=False,
            )
        m.eval()
        prompt = torch.tensor([[4, 5]])
        mask = torch.ones_like(prompt)
        emb, maskfull = m._build_prefix_raw(input_ids_prompt=prompt, mask_prompt=mask)
        self.assertEqual(tuple(emb.shape), (1, 3, 16))
        loss = m(
            labels=torch.tensor([[6, 7]]),
            mask_label=torch.ones(1, 2, dtype=torch.long),
            input_ids_prompt=prompt,
            mask_prompt=mask,
        )
        self.assertTrue(torch.isfinite(loss))

        class Tok:
            def batch_decode(self, ids, **kw):
                return ids.tolist()

        got = m.generate(Tok(), input_ids_prompt=prompt, mask_prompt=mask)
        with torch.inference_mode():
            expected = llm.generate(
                inputs_embeds=emb,
                attention_mask=maskfull,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=0,
            ).tolist()
        self.assertEqual(got, expected)

    def test_all_conditions_abort_missing_images(self):
        from eval_runtime import parser, prepare

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.jsonl"
            p.write_text(
                json.dumps(
                    {
                        "id": "i",
                        "image_id": "missing",
                        "subset": "a",
                        "query": "q",
                        "choices": ["yes", "no"],
                        "answer_index": 0,
                    }
                )
                + "\n"
            )
            for flag in ([], ["--blind"], ["--no-image"]):
                args = parser("cvqa").parse_args(
                    [
                        "--data-path",
                        str(p),
                        "--images-dir",
                        d,
                        "--output-path",
                        d + "/out",
                        "--ckpt",
                        d + "/ckpt",
                        *flag,
                    ]
                )
                with self.assertRaisesRegex(ValueError, "missing original image"):
                    prepare(args)


if __name__ == "__main__":
    unittest.main()
