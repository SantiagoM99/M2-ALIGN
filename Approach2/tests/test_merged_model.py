"""CPU tests for the merged architecture's trainable half, on a stub decoder.

No Qwen, no SigLIP2, no downloads: the point is the mechanics — pooling,
resampling, which positions an early injection touches, that hooks do not
outlive the forward pass, and that nothing frozen acquires a gradient. The two
Qwen-specific assumptions live in `native_visual_positions` and `native_grid`
and are checked against the installed transformers separately.

    python Approach2/tests/test_merged_model.py
"""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "merged"))
from model import (  # noqa: E402
    EARLY_TARGET_LAYERS, DenseVisionExpert, freeze, inject, native_grid,
    native_visual_positions, pool_grid, resample_grid, trainable_parameters,
)

SIDE = 27
PATCHES = SIDE * SIDE
VIS_DIM = 8
LLM_DIM = 16


class StubLayer(nn.Module):
    """Records what it was handed, so a test can assert on the injected input."""

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, hidden_states, **_kwargs):
        self.seen = hidden_states.clone()
        return (hidden_states,)


def dense(batch=2, layers=3):
    return [torch.randn(batch, PATCHES, VIS_DIM) for _ in range(layers)]


class MergedModelTests(unittest.TestCase):
    def test_pooling_reduces_the_grid_and_keeps_the_width(self):
        pooled = pool_grid(torch.randn(2, PATCHES, VIS_DIM), SIDE, 12)
        self.assertEqual(pooled.shape, (2, 144, VIS_DIM))

    def test_pooling_a_constant_grid_preserves_its_value(self):
        flat = torch.full((1, PATCHES, VIS_DIM), 0.5)
        self.assertTrue(torch.allclose(pool_grid(flat, SIDE, 9), torch.full((1, 81, VIS_DIM), 0.5)))

    def test_pooling_is_a_no_op_when_the_target_is_not_smaller(self):
        patches = torch.randn(1, PATCHES, VIS_DIM)
        self.assertTrue(torch.equal(pool_grid(patches, SIDE, SIDE), patches))

    def test_a_wrong_patch_count_is_refused(self):
        with self.assertRaisesRegex(ValueError, "expected 729 patches"):
            pool_grid(torch.randn(1, 100, VIS_DIM), SIDE, 12)

    def test_resampling_hits_the_requested_grid(self):
        out = resample_grid(torch.randn(2, PATCHES, VIS_DIM), SIDE, 5, 7)
        self.assertEqual(out.shape, (2, 35, VIS_DIM))

    def test_resampling_a_constant_grid_preserves_its_value(self):
        flat = torch.full((1, PATCHES, VIS_DIM), -2.0)
        out = resample_grid(flat, SIDE, 4, 6)
        self.assertTrue(torch.allclose(out, torch.full((1, 24, VIS_DIM), -2.0)))

    def test_prefix_mode_emits_pooled_tokens_plus_one_boundary(self):
        expert = DenseVisionExpert(VIS_DIM, LLM_DIM, "prefix", prefix_side=6)
        prefix = expert.forward_prefix(dense(), SIDE)
        self.assertEqual(prefix.shape, (2, 36 + 1, LLM_DIM))

    def test_early_mode_emits_one_addition_per_target_layer(self):
        expert = DenseVisionExpert(VIS_DIM, LLM_DIM, "early")
        additions = expert.layer_additions(dense(), SIDE, 4, 5)
        self.assertEqual(sorted(additions), sorted(EARLY_TARGET_LAYERS))
        for tensor in additions.values():
            self.assertEqual(tensor.shape, (2, 20, LLM_DIM))

    def test_a_wrong_number_of_siglip_layers_is_refused(self):
        expert = DenseVisionExpert(VIS_DIM, LLM_DIM, "early")
        with self.assertRaisesRegex(ValueError, "expected 3 SigLIP2 layers"):
            expert.layer_additions(dense(layers=2), SIDE, 4, 5)

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaisesRegex(ValueError, "mode must be"):
            DenseVisionExpert(VIS_DIM, LLM_DIM, "deepstack")

    def test_injection_touches_only_the_visual_positions(self):
        layers = nn.ModuleList(StubLayer() for _ in range(3))
        hidden = torch.zeros(1, 6, LLM_DIM)
        mask = torch.tensor([[False, True, True, False, False, False]])
        addition = torch.ones(1, 2, LLM_DIM)
        with inject(layers, {0: addition}, mask):
            layers[0](hidden)
        seen = layers[0].seen
        self.assertTrue(torch.equal(seen[0, 1:3], torch.ones(2, LLM_DIM)))
        self.assertTrue(torch.equal(seen[0, [0, 3, 4, 5]], torch.zeros(4, LLM_DIM)))

    def test_injection_adds_rather_than_replaces(self):
        layers = nn.ModuleList([StubLayer()])
        hidden = torch.full((1, 3, LLM_DIM), 2.0)
        mask = torch.tensor([[False, True, False]])
        with inject(layers, {0: torch.full((1, 1, LLM_DIM), 5.0)}, mask):
            layers[0](hidden)
        self.assertTrue(torch.equal(layers[0].seen[0, 1], torch.full((LLM_DIM,), 7.0)))

    def test_a_mask_that_disagrees_with_the_grid_is_refused(self):
        layers = nn.ModuleList([StubLayer()])
        mask = torch.tensor([[True, True, True]])
        with self.assertRaisesRegex(ValueError, "the grid and the mask disagree"):
            with inject(layers, {0: torch.ones(1, 2, LLM_DIM)}, mask):
                layers[0](torch.zeros(1, 3, LLM_DIM))

    def test_replace_mode_overwrites_instead_of_adding(self):
        layers = nn.ModuleList([StubLayer()])
        hidden = torch.full((1, 3, LLM_DIM), 2.0)
        mask = torch.tensor([[True, False, False]])
        with inject(layers, {0: torch.full((1, 1, LLM_DIM), 5.0)}, mask, how="replace"):
            layers[0](hidden)
        self.assertTrue(torch.equal(layers[0].seen[0, 0], torch.full((LLM_DIM,), 5.0)))
        self.assertTrue(torch.equal(layers[0].seen[0, 1], torch.full((LLM_DIM,), 2.0)))

    def test_an_unknown_injection_mode_is_refused(self):
        with self.assertRaisesRegex(ValueError, "how must be"):
            with inject(nn.ModuleList([StubLayer()]), {}, torch.tensor([[True]]), how="scale"):
                pass

    def test_hooks_do_not_outlive_the_context(self):
        layers = nn.ModuleList([StubLayer()])
        mask = torch.tensor([[False, True, False]])
        with inject(layers, {0: torch.ones(1, 1, LLM_DIM)}, mask):
            pass
        self.assertEqual(len(layers[0]._forward_pre_hooks), 0)
        hidden = torch.zeros(1, 3, LLM_DIM)
        layers[0](hidden)
        self.assertTrue(torch.equal(layers[0].seen, hidden))

    def test_only_the_expert_carries_gradients(self):
        expert = DenseVisionExpert(VIS_DIM, LLM_DIM, "prefix", prefix_side=4)
        frozen = freeze(nn.Linear(LLM_DIM, LLM_DIM))
        loss = frozen(expert.forward_prefix(dense(batch=1), SIDE)).sum()
        loss.backward()
        self.assertTrue(all(p.grad is None for p in frozen.parameters()))
        self.assertTrue(any(p.grad is not None for p in expert.parameters()))
        self.assertGreater(trainable_parameters(expert), 0)

    def test_the_boundary_is_initialised_at_zero(self):
        expert = DenseVisionExpert(VIS_DIM, LLM_DIM, "prefix", prefix_side=4)
        self.assertTrue(torch.equal(expert.boundary, torch.zeros(1, 1, LLM_DIM)))

    def test_early_mode_refuses_a_layer_count_it_cannot_pair(self):
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            DenseVisionExpert(VIS_DIM, LLM_DIM, "early", n_layers=4)

    def test_native_helpers_read_ids_and_the_merged_grid(self):
        ids = torch.tensor([[5, 7, 7, 7, 9]])
        self.assertEqual(native_visual_positions(ids, 7).sum().item(), 3)
        self.assertEqual(native_grid(torch.tensor([1, 8, 12]), 2), (4, 6))


if __name__ == "__main__":
    unittest.main()
