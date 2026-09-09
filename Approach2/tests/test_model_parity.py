"""Compare full-prefix numerics to the frozen implementation with tiny towers."""

import copy
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ModelParityTests(unittest.TestCase):
    def test_full_prefix_logits_and_generation_match_freeze(self):
        import torch
        from torch import nn
        from transformers import Gemma2Config, Gemma2ForCausalLM
        from transformers.modeling_outputs import BaseModelOutput
        import model
        from s1_contract import ROOT, SPEC_SHA

        source = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", SPEC_SHA + ":Approach2/model.py"],
            text=True,
        )
        legacy = types.ModuleType("s1_frozen_model")
        exec(compile(source, "<frozen-model>", "exec"), legacy.__dict__)

        class MT(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(d_model=3)
                self.embed = nn.Embedding(32, 3)

            def get_encoder(self):
                return self

            def forward(self, input_ids, **kw):
                return BaseModelOutput(last_hidden_state=self.embed(input_ids))

        class Vision(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(hidden_size=3)

            def forward(self, pixel_values, **kw):
                h = pixel_values.mean((-2, -1)).unsqueeze(1).expand(-1, 4, -1)
                return BaseModelOutput(last_hidden_state=h, hidden_states=(h, h, h))

        torch.manual_seed(13)
        llm = Gemma2ForCausalLM(
            Gemma2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                bos_token_id=2,
                eos_token_id=1,
                pad_token_id=0,
            )
        ).to(torch.bfloat16)
        mt = MT()
        vision = Vision()

        def make(cls):
            with (
                patch(
                    "model.AutoModelForCausalLM.from_pretrained",
                    return_value=copy.deepcopy(llm),
                ),
                patch(
                    "model.M2M100Model.from_pretrained", return_value=copy.deepcopy(mt)
                ),
                patch(
                    "model.AutoModel.from_pretrained",
                    return_value=copy.deepcopy(vision),
                ),
            ):
                return cls("mt", "vis", "llm", 4, 2, 0, vis_layers="1,2,-1").eval()

        old = make(legacy.DualEncoderMerger)
        new = make(model.DualEncoderMerger)
        new.load_state_dict(old.state_dict(), strict=True)
        kw = {
            "input_ids_mt": torch.tensor([[4, 5, 0]]),
            "attention_mask_mt": torch.tensor([[1, 1, 0]]),
            "pixel_values": torch.ones(1, 3, 2, 2),
            "input_ids_prompt": torch.tensor([[6, 7]]),
            "mask_prompt": torch.ones(1, 2, dtype=torch.long),
        }
        labels = torch.tensor([[9, 10]])
        mask = torch.ones_like(labels)

        class Tok:
            def batch_decode(self, ids, **kw):
                return ids.tolist()

        with torch.inference_mode():
            self.assertTrue(
                torch.equal(
                    old(labels=labels, mask_label=mask, **kw),
                    new(labels=labels, mask_label=mask, **kw),
                )
            )
            self.assertEqual(old.generate(Tok(), **kw), new.generate(Tok(), **kw))


if __name__ == "__main__":
    unittest.main()
