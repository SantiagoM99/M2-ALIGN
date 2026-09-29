"""The merged stack: a frozen VLM, a frozen second vision encoder, one mapping.

Composition only. The trainable half and the injection mechanics live in
`model.py`; this file wires them to a real Qwen3-VL and a real SigLIP2 and holds
the one thing neither can decide alone — which positions each mode writes to.

    prefix  the sequence reserves placeholder tokens, and their layer-0 input
            embeddings are replaced by the projected patches. Qwen still builds
            and scatters its own visual tokens, so the native expert is untouched
            and removing this stream leaves exactly Approach 1.
    early   each projected SigLIP2 layer is added into decoder layers 0/1/2 at the
            positions of Qwen's own visual tokens.

**The processor must run at a fixed resolution.** Early injection resamples onto
Qwen's merged grid, and that grid depends on the image's size, so a batch of
differently sized images would need a different resample per example. Pinning
`min_pixels == max_pixels` makes the grid constant, and it also makes every
comparison resolution-matched by construction — the fairness point Maryam raised
on 2026-09-23.

Three Qwen assumptions are isolated and checked rather than trusted:
`native_visual_positions` and `native_grid` in `model.py`, verified by
`verify_qwen_api.py`, and `resolve_decoder_layers` below, which fails naming what
it did find.
"""
from __future__ import annotations

import torch
from torch import nn

from model import (
    DenseVisionExpert, freeze, inject, native_grid, native_visual_positions,
)

LAYER_PATHS = (
    "model.language_model.layers",
    "model.layers",
    "language_model.model.layers",
    "language_model.layers",
)


def resolve_decoder_layers(llm: nn.Module) -> nn.ModuleList:
    """The decoder's layer list, whatever this transformers version calls it."""
    for path in LAYER_PATHS:
        node = llm
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if isinstance(node, nn.ModuleList) and len(node) > 0:
            return node
    available = [name for name, _ in llm.named_children()]
    raise AttributeError(
        f"no decoder layer list at any of {LAYER_PATHS}; the model's children are "
        f"{available}. Add the right path before training, because injecting into "
        "the wrong module would train to a plausible loss and mean nothing"
    )


def dense_features(encoder: nn.Module, pixel_values: torch.Tensor,
                   layers=(9, 18, -1)) -> list[torch.Tensor]:
    """SigLIP2 hidden states at the dense layers, in the order given.

    Same convention as `Approach2/model.py`: a positive index reads
    `hidden_states[i]` and -1 reads `last_hidden_state`. Patch order is identical
    across layers, which is what lets the projections be pooled or resampled
    independently and still describe the same patches.
    """
    out = encoder(pixel_values=pixel_values, output_hidden_states=True)
    return [out.last_hidden_state if i == -1 else out.hidden_states[i] for i in layers]


class MergedVLM(nn.Module):
    """Frozen Qwen3-VL and frozen SigLIP2, joined by one trainable expert."""

    def __init__(self, llm: nn.Module, vis_encoder: nn.Module, expert: DenseVisionExpert,
                 image_token_id: int, spatial_merge_size: int,
                 placeholder_token_id: int | None = None, siglip_side: int = 27,
                 dense_layers=(9, 18, -1)):
        super().__init__()
        if expert.mode == "prefix" and placeholder_token_id is None:
            raise ValueError("prefix mode needs the id of the reserved placeholder token")
        self.llm = freeze(llm)
        self.vis_encoder = freeze(vis_encoder)
        self.expert = expert
        self.image_token_id = image_token_id
        self.spatial_merge_size = spatial_merge_size
        self.placeholder_token_id = placeholder_token_id
        self.siglip_side = siglip_side
        self.dense_layers = dense_layers
        self.decoder_layers = resolve_decoder_layers(self.llm)

    def injection(self, dense, input_ids, image_grid_thw):
        """Where this mode writes, and what it writes there."""
        if self.expert.mode == "prefix":
            prefix = self.expert.forward_prefix(dense, self.siglip_side)
            mask = input_ids == self.placeholder_token_id
            reserved = int(mask[0].sum())
            if reserved != prefix.shape[1]:
                raise ValueError(
                    f"the sequence reserves {reserved} placeholder positions but the "
                    f"prefix is {prefix.shape[1]} tokens; the data builder and "
                    "prefix_side disagree"
                )
            return {0: prefix}, mask, "replace"
        height, width = native_grid(image_grid_thw, self.spatial_merge_size)
        mask = native_visual_positions(input_ids, self.image_token_id)
        return self.expert.layer_additions(dense, self.siglip_side, height, width), mask, "add"

    def forward(self, siglip_pixel_values: torch.Tensor, **qwen_inputs):
        """`qwen_inputs` is whatever the Qwen processor produced, plus `labels`."""
        dense = dense_features(self.vis_encoder, siglip_pixel_values, self.dense_layers)
        features, mask, how = self.injection(
            dense, qwen_inputs["input_ids"], qwen_inputs.get("image_grid_thw")
        )
        with inject(self.decoder_layers, features, mask, how):
            return self.llm(**qwen_inputs)

    def trainable_state_dict(self):
        """Only the expert is ever saved; the three frozen models are not ours."""
        return self.expert.state_dict()
