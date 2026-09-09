"""Approach 2 evaluation on CVQA (multiple-choice, scored open-ended via likelihood).

Mirrors Stage3/evaluate.py's `evaluate_cvqa_open_ended` protocol on the
`parallel` branch: the model is prompted with the question only (no visible
options), and each answer choice is scored by its length-normalized
log-likelihood as a continuation of that prompt — the choice with the
highest mean per-token log-probability wins, and its index is compared
against the gold ``answer_index``. `DualEncoderMerger.forward` already
returns exactly that quantity (mean CE over the label tokens, batch of 1),
so each choice costs one forward pass (~4 per question).

Input JSONL from Stage3/load_evaluation_data.py --benchmark cvqa:
    ``id``, ``query`` (native-language question), ``choices`` (English),
    ``answer_index``, ``source_language``.
Images live at ``<images_dir>/<id>.jpg``.

Usage
-----
    python evaluate_cvqa.py \\
        --data-path ../Stage3/data/cvqa/mn.jsonl \\
        --images-dir ../Stage3/data/cvqa/images \\
        --ckpt ./outputs/stage3/mapping/pytorch_model.bin \\
        --output-path ./outputs/stage3/eval_cvqa_mn.jsonl
"""

from __future__ import annotations


from common import (
    _VQA_SYSTEM,
)


def build_cvqa_open_ended_prompt(question: str) -> str:
    """Must match Stage3/evaluate.py's `build_cvqa_open_ended_prompt`:
    question only, no visible options."""
    return f"Question: {question}"


def format_cvqa_chat(tokenizer_llm, question: str, use_chat_template: bool) -> str:
    """Chat-wrap the CVQA prompt the same way common.format_chat_prompt does
    for open-ended VQA (system folded into the user turn, leading BOS
    stripped) — only the task prompt differs."""
    user_prompt = build_cvqa_open_ended_prompt(question)
    if not use_chat_template or getattr(tokenizer_llm, "chat_template", None) is None:
        return user_prompt
    messages = [{"role": "user", "content": f"{_VQA_SYSTEM}\n\n{user_prompt}"}]
    text = tokenizer_llm.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    bos = getattr(tokenizer_llm, "bos_token", None)
    if bos and text.startswith(bos):
        text = text[len(bos) :]
    return text


def main() -> None:
    from eval_runtime import main as run

    run("cvqa")


if __name__ == "__main__":
    main()
