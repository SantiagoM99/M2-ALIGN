"""CPU tests for the merged stack's wiring, on stub models.

What is checked here is the part neither `model.py` nor a real run can check
cheaply: that each mode writes to the positions it is supposed to write to, that
the layer list is found or the failure names what it found, that a mismatch
between the reserved prefix and the data aborts, and that only the expert is
trainable and only the expert is saved.

    python Approach2/tests/test_merged_bridge.py
"""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

MERGED = Path(__file__).resolve().parents[1] / "merged"
sys.path.insert(0, str(MERGED))
from bridge import MergedVLM, dense_features, resolve_decoder_layers  # noqa: E402
from model import DenseVisionExpert  # noqa: E402

SIDE = 6
PATCHES = SIDE * SIDE
VIS_DIM = 8
LLM_DIM = 16
IMAGE_TOKEN = 900
PLACEHOLDER = 901


class StubLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, hidden_states, **_kwargs):
        self.seen = hidden_states.clone()
        return (hidden_states,)


class StubDecoder(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.layers = nn.ModuleList(StubLayer() for _ in range(n))


class StubLLM(nn.Module):
    """Embeds ids, runs the stub layers, and reports what it was called with."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = StubDecoder()
        self.embed = nn.Embedding(1000, LLM_DIM)
        self.called_with = None

    def forward(self, input_ids=None, **kwargs):
        self.called_with = {"input_ids": input_ids, **kwargs}
        hidden = self.embed(input_ids)
        for layer in self.model.language_model.layers:
            hidden = layer(hidden)[0]
        return hidden


class StubVision(nn.Module):
    """Returns distinguishable hidden states so a test can trace which layer went where."""

    def forward(self, pixel_values=None, output_hidden_states=False):
        b = pixel_values.shape[0]

        class Out:
            hidden_states = [torch.full((b, PATCHES, VIS_DIM), float(i)) for i in range(20)]
            last_hidden_state = torch.full((b, PATCHES, VIS_DIM), -1.0)

        return Out()


def expert(mode, **kw):
    return DenseVisionExpert(VIS_DIM, LLM_DIM, mode, **kw)


class BridgeTests(unittest.TestCase):
    def test_the_layer_list_is_found(self):
        self.assertEqual(len(resolve_decoder_layers(StubLLM())), 3)

    def test_a_model_without_a_layer_list_names_its_children(self):
        class Bare(nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = nn.Linear(2, 2)

        with self.assertRaisesRegex(AttributeError, "children are \\['trunk'\\]"):
            resolve_decoder_layers(Bare())

    def test_dense_features_read_the_requested_layers_in_order(self):
        feats = dense_features(StubVision(), torch.zeros(1, 3, 4, 4), layers=(9, 18, -1))
        self.assertEqual([float(f[0, 0, 0]) for f in feats], [9.0, 18.0, -1.0])

    def test_prefix_mode_replaces_the_reserved_positions_only(self):
        model = MergedVLM(StubLLM(), StubVision(), expert("prefix", prefix_side=2),
                          IMAGE_TOKEN, 2, placeholder_token_id=PLACEHOLDER, siglip_side=SIDE)
        # 2x2 pooled patches plus one boundary = 5 reserved positions
        ids = torch.tensor([[PLACEHOLDER] * 5 + [IMAGE_TOKEN] * 4 + [7, 8]])
        model(torch.zeros(1, 3, 4, 4), input_ids=ids, image_grid_thw=torch.tensor([1, 4, 4]))
        seen = model.decoder_layers[0].seen
        plain = model.llm.embed(ids)
        self.assertFalse(torch.allclose(seen[0, :5], plain[0, :5]))
        self.assertTrue(torch.allclose(seen[0, 5:], plain[0, 5:]))

    def test_prefix_mode_aborts_when_the_reservation_does_not_match(self):
        model = MergedVLM(StubLLM(), StubVision(), expert("prefix", prefix_side=2),
                          IMAGE_TOKEN, 2, placeholder_token_id=PLACEHOLDER, siglip_side=SIDE)
        ids = torch.tensor([[PLACEHOLDER] * 3 + [7]])
        with self.assertRaisesRegex(ValueError, "reserves 3 placeholder positions"):
            model(torch.zeros(1, 3, 4, 4), input_ids=ids, image_grid_thw=torch.tensor([1, 4, 4]))

    def test_early_mode_adds_at_the_native_visual_positions_in_three_layers(self):
        model = MergedVLM(StubLLM(), StubVision(), expert("early"), IMAGE_TOKEN, 2,
                          siglip_side=SIDE)
        ids = torch.tensor([[5] + [IMAGE_TOKEN] * 4 + [6]])
        model(torch.zeros(1, 3, 4, 4), input_ids=ids, image_grid_thw=torch.tensor([1, 4, 4]))
        plain = model.llm.embed(ids)
        for index in range(3):
            seen = model.decoder_layers[index].seen
            self.assertFalse(torch.allclose(seen[0, 1:5], plain[0, 1:5]), f"layer {index}")
        first = model.decoder_layers[0].seen
        self.assertTrue(torch.allclose(first[0, [0, 5]], plain[0, [0, 5]]))

    def test_prefix_mode_needs_a_placeholder_id(self):
        with self.assertRaisesRegex(ValueError, "placeholder token"):
            MergedVLM(StubLLM(), StubVision(), expert("prefix"), IMAGE_TOKEN, 2)

    def test_only_the_expert_is_trainable_and_only_it_is_saved(self):
        model = MergedVLM(StubLLM(), StubVision(), expert("early"), IMAGE_TOKEN, 2,
                          siglip_side=SIDE)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all(n.startswith("expert.") for n in trainable), sorted(trainable))
        self.assertEqual(set(model.trainable_state_dict()), set(model.expert.state_dict()))

    def test_the_inputs_reach_the_llm_unchanged(self):
        model = MergedVLM(StubLLM(), StubVision(), expert("early"), IMAGE_TOKEN, 2,
                          siglip_side=SIDE)
        ids = torch.tensor([[IMAGE_TOKEN] * 4])
        model(torch.zeros(1, 3, 4, 4), input_ids=ids, image_grid_thw=torch.tensor([1, 4, 4]),
              attention_mask=torch.ones(1, 4, dtype=torch.long))
        self.assertIn("attention_mask", model.llm.called_with)
        self.assertTrue(torch.equal(model.llm.called_with["input_ids"], ids))


if __name__ == "__main__":
    unittest.main()
