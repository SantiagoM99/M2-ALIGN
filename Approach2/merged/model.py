"""A second frozen visual expert for a frozen VLM, in two injection modes.

The merged architecture keeps Qwen3-VL's native vision tower and DeepStack
exactly as they are and adds SigLIP2's dense layers 9/18/-1 as a second expert
through a trainable mapping. The motivation is measured, not argued: on identical
items the two towers win on different image distributions — Qwen's native path
reaches dV +29.80 on xGQA against our +17.39, and ours +10.23 on CVQA against its
+7.00 (DESIGN 2026-09-20). Nothing frozen is ever updated; the mapping is the
only trainable part.

Where the second expert enters is a measured variable rather than a guess, and
ANCHOR (arXiv 2608.15085) predicts the answer: it argues non-English visual
reasoning degrades because text is mapped into an English semantic space in early
layers while visual representations have not yet matured, so vision is
functionally invisible exactly when the translation happens. If that holds, early
injection should beat a prefix.

    prefix  the projected patches become soft tokens before the chat template.
            Qwen's forward is untouched, and removing the stream leaves exactly
            Approach 1, so the ablation is a single variable by construction.
    early   each SigLIP2 layer is projected and added into one of the first three
            decoder layers at the positions of the native visual tokens, after
            bilinear resampling onto Qwen's merged grid. This mirrors what
            DenseConnector and DeepStack do, with a second encoder in parallel.

No gate on either path. A trained zero-initialised prefix gate was measured and
rejected in this project (DESIGN D7, xGQA 19.41), so the streams are combined by
plain addition and the question of whether the LLM ignores a redundant stream is
answered by the ablation instead of hidden behind a learned scalar.

Everything Qwen-specific lives in `native_visual_positions` and `native_grid`,
which are the two assumptions to verify against the installed transformers
before a training run.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import nn

SIGLIP_LAYERS = (9, 18, -1)
EARLY_TARGET_LAYERS = (0, 1, 2)


class MLP(nn.Module):
    """Two-layer projection, the same shape both approaches already use."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def pool_grid(patches: torch.Tensor, side: int, target_side: int) -> torch.Tensor:
    """Average-pool a square patch grid down to `target_side` per side.

    SigLIP2 at 384 gives 27x27 = 729 patches, which as a prefix would cost more
    context than Qwen's own visual block. Pooling is parameter-free on purpose:
    a learned pooler would be a second thing to train and a second explanation
    for any gain.
    """
    b, n, d = patches.shape
    if n != side * side:
        raise ValueError(f"expected {side * side} patches, got {n}")
    if target_side >= side:
        return patches
    grid = patches.transpose(1, 2).reshape(b, d, side, side)
    pooled = F.adaptive_avg_pool2d(grid, (target_side, target_side))
    return pooled.flatten(2).transpose(1, 2)


def resample_grid(patches: torch.Tensor, side: int, height: int, width: int) -> torch.Tensor:
    """Bilinearly resample a square patch grid onto Qwen's merged (h, w) grid.

    Additive injection needs one vector per native visual token, and SigLIP2's
    grid is fixed at 384 while Qwen's depends on the image's own resolution, so
    the two only line up after resampling.
    """
    b, n, d = patches.shape
    if n != side * side:
        raise ValueError(f"expected {side * side} patches, got {n}")
    grid = patches.transpose(1, 2).reshape(b, d, side, side)
    out = F.interpolate(grid, size=(height, width), mode="bilinear", align_corners=False)
    return out.flatten(2).transpose(1, 2)


def native_visual_positions(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Boolean mask of the LLM positions holding Qwen's own visual tokens.

    ASSUMPTION to verify against the installed transformers: Qwen3-VL expands one
    `<|image_pad|>` id per merged visual token, so counting that id gives the
    native grid's token count.
    """
    return input_ids == image_token_id


def native_grid(image_grid_thw: torch.Tensor, spatial_merge_size: int) -> tuple[int, int]:
    """Qwen's merged visual grid for a single image, as (height, width).

    ASSUMPTION to verify: the processor returns `image_grid_thw` as (t, h, w) in
    pre-merge patch units, and the model merges `spatial_merge_size` in each
    spatial direction.
    """
    _, h, w = (int(v) for v in image_grid_thw.reshape(-1)[:3])
    return h // spatial_merge_size, w // spatial_merge_size


class DenseVisionExpert(nn.Module):
    """The trainable half: SigLIP2 dense features into the LLM's hidden space.

    Holds no frozen weights. `llm_dim` is the decoder's hidden size, `vis_dim`
    SigLIP2's per-layer width, and `mode` decides whether `forward_prefix` or
    `layer_additions` is the one that gets used.
    """

    def __init__(self, vis_dim: int, llm_dim: int, mode: str,
                 n_layers: int = len(SIGLIP_LAYERS), prefix_side: int = 12):
        super().__init__()
        if mode not in ("prefix", "early"):
            raise ValueError(f"mode must be 'prefix' or 'early', not {mode!r}")
        if mode == "early" and n_layers != len(EARLY_TARGET_LAYERS):
            raise ValueError(
                f"early injection pairs one SigLIP2 layer with each of decoder layers "
                f"{EARLY_TARGET_LAYERS}, so it needs exactly {len(EARLY_TARGET_LAYERS)} "
                f"layers, not {n_layers}"
            )
        self.mode = mode
        self.prefix_side = prefix_side
        self.n_layers = n_layers
        if mode == "prefix":
            self.project = MLP(vis_dim * n_layers, llm_dim)
        else:
            self.project = nn.ModuleList(MLP(vis_dim, llm_dim) for _ in range(n_layers))
        self.boundary = nn.Parameter(torch.zeros(1, 1, llm_dim))

    def forward_prefix(self, dense: list[torch.Tensor], side: int) -> torch.Tensor:
        """Soft tokens for the prefix mode: pooled patches plus one boundary."""
        if len(dense) != self.n_layers:
            raise ValueError(f"expected {self.n_layers} SigLIP2 layers, got {len(dense)}")
        merged = self.project(torch.cat(dense, dim=-1))
        pooled = pool_grid(merged, side, self.prefix_side)
        boundary = self.boundary.expand(pooled.shape[0], -1, -1)
        return torch.cat([pooled, boundary], dim=1)

    def layer_additions(self, dense: list[torch.Tensor], side: int,
                        height: int, width: int) -> dict[int, torch.Tensor]:
        """One addition per target decoder layer, on Qwen's own visual grid.

        SigLIP2's shallow, middle and final layers go to decoder layers 0, 1 and
        2 respectively: the same shallow-to-early correspondence DenseConnector
        and DeepStack use, so the ordering is a choice with a precedent rather
        than an arbitrary pairing.
        """
        if len(dense) != self.n_layers:
            raise ValueError(f"expected {self.n_layers} SigLIP2 layers, got {len(dense)}")
        out = {}
        for index, (features, target) in enumerate(zip(dense, EARLY_TARGET_LAYERS)):
            projected = self.project[index](features)
            out[target] = resample_grid(projected, side, height, width)
        return out


@contextlib.contextmanager
def inject(decoder_layers, features: dict[int, torch.Tensor], mask: torch.Tensor,
           how: str = "add"):
    """Write `features[i]` into decoder layer i's input at the masked positions.

    Both injection modes go through here, which is why `how` exists.

    `add` is early injection: the features are added to Qwen's own visual tokens,
    so the native expert keeps its content and ours is a correction on top.

    `replace` is the prefix mode. The prefix cannot be prepended to
    `inputs_embeds`, because Qwen builds its visual embeddings inside its own
    forward from `pixel_values` and scatters them at the image-pad positions —
    handing it ready-made embeddings turns the native expert off, and the native
    expert is the other half of the architecture. So the sequence reserves
    placeholder tokens, Qwen embeds everything including its scatter, and their
    embeddings are replaced here. Positions and rope stay Qwen's own.

    A context manager because a hook that outlives the forward pass would apply
    stale features to the next batch, which is the kind of bug that produces a
    plausible loss curve and a meaningless model.
    """
    if how not in ("add", "replace"):
        raise ValueError(f"how must be 'add' or 'replace', not {how!r}")
    handles = []

    def make_hook(addition):
        def hook(_module, args, kwargs):
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            patched = hidden.clone()
            for b in range(hidden.shape[0]):
                positions = mask[b].nonzero(as_tuple=True)[0]
                if positions.numel() != addition.shape[1]:
                    raise ValueError(
                        f"{positions.numel()} masked positions but {addition.shape[1]} "
                        "features: the grid and the mask disagree"
                    )
                incoming = addition[b].to(hidden.dtype)
                patched[b, positions] = (
                    hidden[b, positions] + incoming if how == "add" else incoming
                )
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = patched
                return args, kwargs
            return (patched,) + tuple(args[1:]), kwargs

        return hook

    try:
        for index, addition in features.items():
            handles.append(
                decoder_layers[index].register_forward_pre_hook(make_hook(addition), with_kwargs=True)
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


def trainable_parameters(expert: DenseVisionExpert) -> int:
    return sum(p.numel() for p in expert.parameters() if p.requires_grad)


def freeze(module: nn.Module) -> nn.Module:
    """Freeze everything in `module`; the merged stack trains mappings only."""
    for p in module.parameters():
        p.requires_grad = False
    module.eval()
    return module
