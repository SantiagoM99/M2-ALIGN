"""Approach 2 model: frozen NLLB + frozen SigLIP 2 → trainable mappings → frozen text-only LLM.

Unlike Stage1-3 (Approach 1), which bolt a multilingual encoder onto Qwen3-VL
(a model that already sees), Approach 2 composes a multilingual VLM out of
three frozen parts:

    - NLLB-200 encoder      → language coverage   (frozen)
    - SigLIP 2 vision tower → visual grounding    (frozen)
    - Gemma 2 (text-only)   → reasoning/generation (frozen)

Only the two Mapping layers (and their boundary embeddings) are trained.
The LLM input prefix is built as:

    ``[BOS] + X_f + [b_txt] + V_f + [b_vis] + T``

where:
    ``X_f``   = ``mapping_txt(encoder_mt(query))``          (trainable path)
    ``V_f``   = ``mapping_vis(τ_k(encoder_vis(image)))``    (trainable path)
    ``b_*``   = learnable per-branch boundary embeddings
    ``T``     = LLM token embedding of the prompt template  (frozen path)

Either branch can be disabled, so this single class serves all stages:
    Stage 1 (text mapping):   use_vision_branch=False → ``[BOS] + X_f + b_txt``
    Stage 2 (vision mapping): use_text_branch=False   → ``[BOS] + V_f + b_vis``
    Stage 3 (joint VQA):      both branches on        → full prefix
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, M2M100Model
from transformers.generation.logits_process import LogitsProcessorList


def _squeeze_pad(
    hidden_states: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove padding columns that are zero for every example in the batch.

    Identical to the helper in Stage2/Stage3 so all approaches share the
    same sequence-packing semantics (pads sorted to the front, then all-pad
    columns dropped).

    Args:
        hidden_states: Float tensor of shape ``[batch, seq, dim]``.
        masks: Long tensor of shape ``[batch, seq]`` (1 = real, 0 = pad).

    Returns:
        ``(hidden_states, masks, keep_idx)`` with padding columns removed.
        ``keep_idx`` is a boolean mask over the original sequence positions.
    """
    x_01 = (masks != 0).long()
    seq_len = x_01.size(1)
    offset = (
        torch.arange(1, seq_len + 1, dtype=torch.long, device=x_01.device)
        .unsqueeze(0)
        .expand_as(x_01)
    )
    x_01 = x_01 * offset
    _, idx = x_01.sort(1, descending=False)
    masks = masks.gather(1, idx)
    idx_ex = idx.unsqueeze(-1).expand_as(hidden_states)
    hidden_states = hidden_states.gather(1, idx_ex)
    bs, _, dim = hidden_states.size()
    masks_sum = masks.sum(dim=0)
    keep_idx = (masks_sum > 0).unsqueeze(0).expand_as(masks)
    masks = masks[keep_idx].view(bs, -1)
    hidden_states = hidden_states[keep_idx.unsqueeze(-1).expand_as(hidden_states)].view(bs, -1, dim)
    return hidden_states, masks, keep_idx


class PresencePenaltyGeneratedOnly:
    """Logits processor that penalises tokens seen only in the generated suffix.

    Mirrors the Stage 1-3 processor so all approaches decode consistently.

    Args:
        penalty: Magnitude of the penalty to subtract from repeated logits.
        prompt_len: Number of tokens in the conditioning prefix.
    """

    def __init__(self, penalty: float, prompt_len: int) -> None:
        self.penalty = float(penalty)
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0:
            return scores
        if input_ids.size(1) <= self.prompt_len:
            return scores
        gen_part = input_ids[:, self.prompt_len:]
        for b in range(input_ids.size(0)):
            seen = torch.unique(gen_part[b])
            scores[b, seen] -= self.penalty
        return scores


class MLP(nn.Module):
    """Two-layer MLP: Linear(in_dim, in_dim*2) → ReLU → Linear(in_dim*2, llm_dim).

    Args:
        in_dim: Input dimension (encoder hidden size).
        llm_dim: Output dimension (LLM embedding size).
    """

    def __init__(self, in_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, in_dim * 2)
        self.linear2 = nn.Linear(in_dim * 2, llm_dim)
        self.relu = nn.ReLU()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project *hidden_state* into LLM embedding space.

        Args:
            hidden_state: Float tensor of shape ``[batch, seq, in_dim]``.

        Returns:
            Float tensor of shape ``[batch, seq, llm_dim]``.
        """
        return self.linear2(self.relu(self.linear1(hidden_state)))


class Mapping(nn.Module):
    """Trainable mapping: MLP projection + learnable end-boundary token.

    Same class as in Stage1-3 so checkpoint format (``mlp.*`` +
    ``end_boundary``) stays interchangeable across approaches.

    Args:
        in_dim: Encoder hidden-state dimension.
        llm_dim: LLM embedding dimension.
    """

    def __init__(self, in_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.mlp = MLP(in_dim, llm_dim)
        self.end_boundary = nn.Parameter(torch.zeros(1, 1, llm_dim), requires_grad=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the MLP projection to *hidden_states*."""
        return self.mlp(hidden_states)

    def get_embed(self) -> torch.Tensor:
        """Return the learnable end-boundary embedding ``[1, 1, llm_dim]``."""
        return self.end_boundary


class DualEncoderMerger(nn.Module):
    """Frozen NLLB + frozen SigLIP 2 + frozen text-only LLM, trainable mappings.

    Args:
        mt_path: HF id or local path for the NLLB/M2M MT model. Ignored when
            *use_text_branch* is ``False``.
        vis_path: HF id or local path for the SigLIP/SigLIP2 checkpoint (the
            vision tower is extracted automatically from a dual-tower model).
            Ignored when *use_vision_branch* is ``False``.
        llm_path: HF id or local path for the text-only LLM (Gemma 2).
        max_gen_len: Maximum new tokens at inference.
        llm_bos_token_id: LLM BOS token id; falls back to pad when ``None``.
        llm_pad_token_id: LLM PAD token id.
        use_text_branch: Build/run the NLLB → mapping_txt branch.
        use_vision_branch: Build/run the SigLIP → mapping_vis branch.
        max_vis_tokens: τ_k token selector — keep only the first k visual
            tokens (0 keeps all). Applied before the mapping MLP, which is
            equivalent to applying it after (the MLP is per-token) but
            cheaper.
        local_files_only: Skip Hub downloads when ``True``.
    """

    def __init__(
        self,
        mt_path: str | None,
        vis_path: str | None,
        llm_path: str,
        max_gen_len: int,
        llm_bos_token_id: int | None,
        llm_pad_token_id: int | None,
        use_text_branch: bool = True,
        use_vision_branch: bool = True,
        max_vis_tokens: int = 0,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if not use_text_branch and not use_vision_branch:
            raise ValueError("At least one of use_text_branch/use_vision_branch must be True.")
        self.max_gen_len = max_gen_len
        self.use_text_branch = use_text_branch
        self.use_vision_branch = use_vision_branch
        self.max_vis_tokens = max_vis_tokens

        # Frozen text-only LLM (bf16 to keep Gemma2-9B within a single GPU).
        self.model_llm = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=torch.bfloat16, local_files_only=local_files_only
        )
        for p in self.model_llm.parameters():
            p.requires_grad = False
        self.llm_embedding_layer = self.model_llm.get_input_embeddings()
        llm_dim = getattr(
            self.llm_embedding_layer, "embedding_dim", self.llm_embedding_layer.weight.shape[1]
        )

        # Frozen NLLB encoder + trainable text mapping.
        self.model_mt = None
        self.encoder_mt = None
        self.mapping_txt = None
        if use_text_branch:
            if mt_path is None:
                raise ValueError("mt_path is required when use_text_branch=True.")
            self.model_mt = M2M100Model.from_pretrained(mt_path, local_files_only=local_files_only)
            self.encoder_mt = self.model_mt.get_encoder()
            for p in self.model_mt.parameters():
                p.requires_grad = False
            self.mapping_txt = Mapping(self.model_mt.config.d_model, llm_dim)

        # Frozen SigLIP vision tower + trainable vision mapping.
        self.encoder_vis = None
        self.mapping_vis = None
        if use_vision_branch:
            if vis_path is None:
                raise ValueError("vis_path is required when use_vision_branch=True.")
            vis_model = AutoModel.from_pretrained(vis_path, local_files_only=local_files_only)
            # Dual-tower SigLIP checkpoints expose .vision_model; keep only that
            # so the text tower is released.
            self.encoder_vis = getattr(vis_model, "vision_model", vis_model)
            for p in self.encoder_vis.parameters():
                p.requires_grad = False
            self.mapping_vis = Mapping(self.encoder_vis.config.hidden_size, llm_dim)

        self.llm_pad_token_id = llm_pad_token_id
        self.llm_bos_token_id = llm_bos_token_id if llm_bos_token_id is not None else llm_pad_token_id
        if self.llm_bos_token_id is None:
            raise ValueError("Need at least one of llm_bos_token_id or llm_pad_token_id.")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M total")

    @property
    def llm_dtype(self) -> torch.dtype:
        """Dtype of the frozen LLM weights (bf16). Mappings stay fp32 for
        stable AdamW updates; their outputs are cast to this dtype."""
        return self.llm_embedding_layer.weight.dtype

    def _encode_text(
        self,
        input_ids_mt: torch.Tensor,
        attention_mask_mt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run NLLB encoder + text mapping, append the text boundary.

        Args:
            input_ids_mt: NLLB token ids ``[B, mt_seq]``.
            attention_mask_mt: NLLB attention mask ``[B, mt_seq]``.

        Returns:
            ``(embeds, mask)`` of shapes ``[B, mt_seq+1, llm_dim]`` and
            ``[B, mt_seq+1]`` (``X_f + b_txt``).
        """
        bs = input_ids_mt.size(0)
        dtype = self.llm_dtype
        mt_out = self.encoder_mt(
            input_ids=input_ids_mt,
            attention_mask=attention_mask_mt,
            output_hidden_states=False,
        )
        x_f = self.mapping_txt(mt_out[0]).to(dtype)
        b_txt = self.mapping_txt.get_embed().expand(bs, 1, -1).to(dtype)
        ones = torch.ones(bs, 1, dtype=torch.long, device=input_ids_mt.device)
        return torch.cat([x_f, b_txt], dim=1), torch.cat([attention_mask_mt, ones], dim=1)

    def _encode_vision(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run SigLIP vision tower + vision mapping, append the vision boundary.

        SigLIP at fixed resolution yields a constant patch count per image,
        so no padding/masking is needed across the batch.

        Args:
            pixel_values: Preprocessed images ``[B, 3, H, W]``.

        Returns:
            ``(embeds, mask)`` of shapes ``[B, k+1, llm_dim]`` and
            ``[B, k+1]`` (``V_f + b_vis``), where k is the (possibly
            τ_k-truncated) visual token count.
        """
        bs = pixel_values.size(0)
        dtype = self.llm_dtype
        vis_hidden = self.encoder_vis(pixel_values=pixel_values).last_hidden_state
        if self.max_vis_tokens > 0:
            vis_hidden = vis_hidden[:, : self.max_vis_tokens]
        v_f = self.mapping_vis(vis_hidden.float()).to(dtype)
        b_vis = self.mapping_vis.get_embed().expand(bs, 1, -1).to(dtype)
        mask = torch.ones(bs, v_f.size(1) + 1, dtype=torch.long, device=pixel_values.device)
        return torch.cat([v_f, b_vis], dim=1), mask

    def _build_prefix_raw(
        self,
        input_ids_mt: torch.Tensor | None = None,
        attention_mask_mt: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        input_ids_prompt: torch.Tensor | None = None,
        mask_prompt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble ``[BOS] (+ X_f + b_txt) (+ V_f + b_vis) (+ T)`` without squeeze_pad.

        Which segments appear is determined by which inputs are passed (and
        by the branch toggles set at construction).

        Returns:
            ``(llm_embeds, llm_mask)`` of shapes ``[B, prefix_len, llm_dim]``
            and ``[B, prefix_len]``.
        """
        if input_ids_mt is not None:
            ref = input_ids_mt
        elif pixel_values is not None:
            ref = pixel_values
        else:
            raise ValueError("Need at least one of input_ids_mt or pixel_values.")
        bs = ref.size(0)
        device = ref.device
        dtype = self.llm_dtype

        bos = torch.full((bs,), self.llm_bos_token_id, dtype=torch.long, device=device)
        segments = [self.llm_embedding_layer(bos).view(bs, 1, -1).to(dtype)]
        seg_masks = [torch.ones(bs, 1, dtype=torch.long, device=device)]

        if input_ids_mt is not None:
            if not self.use_text_branch:
                raise ValueError("Received text inputs but use_text_branch=False.")
            emb, mask = self._encode_text(input_ids_mt, attention_mask_mt)
            segments.append(emb)
            seg_masks.append(mask)

        if pixel_values is not None:
            if not self.use_vision_branch:
                raise ValueError("Received pixel_values but use_vision_branch=False.")
            emb, mask = self._encode_vision(pixel_values)
            segments.append(emb)
            seg_masks.append(mask)

        if input_ids_prompt is not None:
            segments.append(self.llm_embedding_layer(input_ids_prompt).to(dtype))
            seg_masks.append(mask_prompt)

        return torch.cat(segments, dim=1), torch.cat(seg_masks, dim=1)

    def forward(
        self,
        labels: torch.Tensor,
        mask_label: torch.Tensor,
        input_ids_mt: torch.Tensor | None = None,
        attention_mask_mt: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        input_ids_prompt: torch.Tensor | None = None,
        mask_prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the teacher-forcing cross-entropy loss.

        Args:
            labels: LLM token ids for the target, shape ``[B, label_seq]``.
            mask_label: Attention mask for *labels*.
            input_ids_mt: NLLB token ids ``[B, mt_seq]`` (text branch).
            attention_mask_mt: NLLB attention mask (text branch).
            pixel_values: Preprocessed images ``[B, 3, H, W]`` (vision branch).
            input_ids_prompt: Optional LLM token ids for the prompt ``T``.
            mask_prompt: Attention mask for *input_ids_prompt*.

        Returns:
            Scalar cross-entropy loss tensor.
        """
        bs = labels.size(0)
        dtype = self.llm_dtype

        llm_embeds, llm_mask = self._build_prefix_raw(
            input_ids_mt, attention_mask_mt, pixel_values, input_ids_prompt, mask_prompt
        )

        pad_labels = torch.full_like(llm_mask, -100)
        label_embedding = self.llm_embedding_layer(labels).to(dtype)
        llm_embeds = torch.cat([llm_embeds, label_embedding], dim=1)
        llm_mask = torch.cat([llm_mask, mask_label], dim=1)
        labels_masked = labels * mask_label + (-100) * (1 - mask_label)
        labels_full = torch.cat([pad_labels, labels_masked], dim=1)

        llm_embeds, llm_mask, cut_idx = _squeeze_pad(llm_embeds, llm_mask)
        labels_full = labels_full[cut_idx].view(bs, -1)

        out = self.model_llm(
            inputs_embeds=llm_embeds,
            attention_mask=llm_mask,
            labels=labels_full,
        )
        return out.loss

    @torch.inference_mode()
    def generate(
        self,
        tokenizer_llm,
        input_ids_mt: torch.Tensor | None = None,
        attention_mask_mt: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        input_ids_prompt: torch.Tensor | None = None,
        mask_prompt: torch.Tensor | None = None,
        generation_kwargs: dict | None = None,
        presence_penalty: float | None = None,
    ) -> list[str]:
        """Greedy-decode continuations from the assembled prefix.

        Args:
            tokenizer_llm: LLM tokenizer used for decoding.
            input_ids_mt: NLLB token ids (text branch).
            attention_mask_mt: NLLB attention mask (text branch).
            pixel_values: Preprocessed images (vision branch).
            input_ids_prompt: Optional LLM prompt token ids ``T``.
            mask_prompt: Attention mask for *input_ids_prompt*.
            generation_kwargs: Extra kwargs forwarded to ``model_llm.generate``.
            presence_penalty: If non-zero, apply
                :class:`PresencePenaltyGeneratedOnly`.

        Returns:
            List of decoded strings, one per batch row.
        """
        llm_embeds, llm_mask = self._build_prefix_raw(
            input_ids_mt, attention_mask_mt, pixel_values, input_ids_prompt, mask_prompt
        )
        llm_embeds, llm_mask, _ = _squeeze_pad(llm_embeds, llm_mask)
        prefix_len = llm_embeds.size(1)

        gen_kw: dict = dict(
            inputs_embeds=llm_embeds,
            attention_mask=llm_mask,
            max_new_tokens=self.max_gen_len,
            pad_token_id=self.llm_pad_token_id,
            do_sample=False,
        )
        if presence_penalty is not None and presence_penalty != 0.0:
            gen_kw["logits_processor"] = LogitsProcessorList(
                [PresencePenaltyGeneratedOnly(presence_penalty, prefix_len)]
            )
        if generation_kwargs:
            gen_kw.update(generation_kwargs)

        ids = self.model_llm.generate(**gen_kw)
        # Some HF versions return only new tokens when using inputs_embeds.
        new_ids = ids[:, prefix_len:] if ids.shape[1] > prefix_len else ids
        return tokenizer_llm.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
