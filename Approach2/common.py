"""Shared helpers for Approach 2 training/evaluation scripts.

Consolidates the tokenisation, checkpoint, W&B and prompt utilities that
Stage1-3 duplicate per stage, so the three Approach 2 scripts stay small.
"""
from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime

import torch

try:
    import wandb
except ImportError:
    wandb = None


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, name: str) -> logging.Logger:
    """Create a logger writing to stdout and a timestamped file in *log_dir*."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{ts}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def mt_input_features(
    texts: list[str],
    nllb_lang_tags: list[str],
    tokenizer_mt,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise multilingual texts with the NLLB tokenizer.

    Sets ``src_lang`` per example from the per-row NLLB language tag so the
    correct language-tag token is inserted.

    Args:
        texts: Source-language strings.
        nllb_lang_tags: NLLB tags parallel to *texts* (e.g. ``"fra_Latn"``).
        tokenizer_mt: An instantiated NLLB tokenizer.
        max_seq_len: Maximum token length; longer strings are right-truncated.
        device: Target device.

    Returns:
        ``(input_ids, attention_mask)`` of shape ``[B, seq_len]``.
    """
    ids, masks = [], []
    for text, tag in zip(texts, nllb_lang_tags):
        tokenizer_mt.src_lang = tag
        enc = tokenizer_mt(text, truncation=True, max_length=max_seq_len, padding=False)
        ids.append(enc["input_ids"])
        masks.append(enc["attention_mask"])

    max_len = max(len(x) for x in ids)
    pad_id = tokenizer_mt.pad_token_id
    for i in range(len(ids)):
        while len(ids[i]) < max_len:
            ids[i].append(pad_id)
            masks[i].append(0)

    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def llm_input_features(
    texts: list[str],
    tokenizer_llm,
    max_seq_len: int,
    add_bos: bool,
    add_eos: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise strings with the LLM tokenizer, controlling BOS/EOS insertion.

    Args:
        texts: Strings to tokenize.
        tokenizer_llm: LLM tokenizer (Gemma).
        max_seq_len: Maximum token length.
        add_bos: Whether to prepend BOS.
        add_eos: Whether to append EOS.
        device: Target device.

    Returns:
        ``(input_ids, attention_mask)`` of shape ``[B, seq_len]``.
    """
    tokenizer_llm.add_bos_token = add_bos
    tokenizer_llm.add_eos_token = add_eos
    enc = tokenizer_llm(
        texts,
        truncation=True,
        max_length=max_seq_len,
        padding=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_VQA_SYSTEM = "You are a helpful assistant that answers questions about images."


def build_open_ended_prompt(question: str) -> str:
    """Task prompt for open-ended VQA — must match Stage3/evaluate.py on the
    `parallel` branch. "in English" pins the answer language so exact match
    doesn't punish correct answers given in the question's language."""
    return f"Question: {question}\nAnswer with a single word or short phrase, in English."


def format_chat_prompt(tokenizer_llm, question: str, use_chat_template: bool) -> str:
    """Format the LLM-side query term ``T`` for the given tokenizer.

    Gemma 2 instruction-tuned checkpoints reject a ``system`` role and
    prepend ``<bos>`` inside the template, so the system text is folded into
    the user turn and any leading BOS is stripped (the model prepends its
    own BOS embedding at the start of the prefix).

    Args:
        tokenizer_llm: LLM tokenizer (must expose ``apply_chat_template``
            when *use_chat_template* is ``True``).
        question: Raw question text.
        use_chat_template: When ``False``, return the bare task prompt
            (appropriate for non-instruction-tuned base models).

    Returns:
        The formatted prompt string.
    """
    user_prompt = build_open_ended_prompt(question)
    if not use_chat_template or getattr(tokenizer_llm, "chat_template", None) is None:
        return user_prompt
    messages = [{"role": "user", "content": f"{_VQA_SYSTEM}\n\n{user_prompt}"}]
    text = tokenizer_llm.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    bos = getattr(tokenizer_llm, "bos_token", None)
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return text


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_mapping_checkpoint(path: str, model, step: int, loss: float) -> None:
    """Save the trainable mapping weights (whichever branches exist).

    The per-branch state dicts use the same ``Mapping`` layout as Stage1-3
    checkpoints, so a single branch can be warm-started from either an
    Approach 2 or an Approach 1 checkpoint (dimensions permitting).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: dict = {"step": step, "loss": loss}
    if model.mapping_txt is not None:
        payload["mapping_txt"] = model.mapping_txt.state_dict()
    if model.mapping_vis is not None:
        payload["mapping_vis"] = model.mapping_vis.state_dict()
    torch.save(payload, path)


def load_mapping_checkpoint(path: str, model, logger=None) -> None:
    """Load mapping weights into whichever branches the checkpoint covers.

    Accepts Approach 2 checkpoints (``mapping_txt`` / ``mapping_vis`` keys)
    and legacy Approach 1 checkpoints (``model_state_dict`` — interpreted as
    the text mapping).
    """
    ckpt = torch.load(path, map_location="cpu")
    loaded = []
    if "mapping_txt" in ckpt and model.mapping_txt is not None:
        model.mapping_txt.load_state_dict(ckpt["mapping_txt"])
        loaded.append("mapping_txt")
    if "mapping_vis" in ckpt and model.mapping_vis is not None:
        model.mapping_vis.load_state_dict(ckpt["mapping_vis"])
        loaded.append("mapping_vis")
    if "model_state_dict" in ckpt and model.mapping_txt is not None and not loaded:
        model.mapping_txt.load_state_dict(ckpt["model_state_dict"], strict=False)
        loaded.append("mapping_txt (legacy Approach 1 format)")
    if not loaded:
        raise ValueError(f"No loadable mapping weights found in {path} (keys: {list(ckpt)})")
    if logger is not None:
        logger.info("Loaded %s from %s", ", ".join(loaded), path)


def save_training_state(
    path: str,
    model,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    best_val: float,
) -> None:
    """Save a resumable snapshot (weights + optimizer + RNG + progress).

    Written atomically (tmp file + rename) so a mid-write SLURM kill can't
    leave a corrupt checkpoint behind.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    payload = {
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "best_val": best_val,
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "python_rng_state": random.getstate(),
    }
    if model.mapping_txt is not None:
        payload["mapping_txt"] = model.mapping_txt.state_dict()
    if model.mapping_vis is not None:
        payload["mapping_vis"] = model.mapping_vis.state_dict()
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_training_state(
    path: str,
    model,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, int, float]:
    """Restore a snapshot written by :func:`save_training_state`."""
    ckpt = torch.load(path, map_location=device)
    if "mapping_txt" in ckpt and model.mapping_txt is not None:
        model.mapping_txt.load_state_dict(ckpt["mapping_txt"])
    if "mapping_vis" in ckpt and model.mapping_vis is not None:
        model.mapping_vis.load_state_dict(ckpt["mapping_vis"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    random.setstate(ckpt["python_rng_state"])
    return ckpt["epoch"], ckpt["step_in_epoch"], ckpt["global_step"], ckpt["best_val"]


# ---------------------------------------------------------------------------
# W&B (same behaviour as Stage 2/3 scripts)
# ---------------------------------------------------------------------------

def _load_wandb_key_from_tokens() -> bool:
    for candidate in (
        os.path.join(os.getcwd(), ".tokens"),
        os.path.join(os.path.dirname(os.getcwd()), ".tokens"),
    ):
        if not os.path.isfile(candidate):
            continue
        with open(candidate, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "WANDB_API_KEY":
                    value = value.strip().strip('"').strip("'")
                    if value:
                        os.environ["WANDB_API_KEY"] = value
                        return True
    return False


def init_wandb_or_disable(args, config: dict) -> bool:
    """Initialise W&B with online/offline fallback; never raises."""
    if not getattr(args, "use_wandb", False) or wandb is None:
        return False
    mode = args.wandb_mode.lower()
    if mode in {"auto", "online"} and not os.environ.get("WANDB_API_KEY"):
        _load_wandb_key_from_tokens()
    if mode == "offline" or (mode == "auto" and not os.environ.get("WANDB_API_KEY")):
        os.environ["WANDB_MODE"] = "offline"
    try:
        if os.environ.get("WANDB_API_KEY"):
            wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            config=config,
        )
        return True
    except Exception as exc:
        print(f"wandb init failed ({exc}); disabling.")
        return False
