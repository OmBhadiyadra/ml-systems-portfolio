"""
Evaluation: compares the LoRA fine-tuned model against the base model.

Reports:
  - Perplexity on a held-out split (base vs. fine-tuned)
  - Trainable parameter count / percentage from the LoRA adapter

Usage:
    python evaluate.py --config config.yaml --adapter_dir outputs/lora_run/final_adapter
"""

import argparse
import math

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts, device, max_length=256):
    model.eval()
    losses = []
    for text in texts:
        enc = tokenizer(
            text, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        if enc["input_ids"].shape[1] < 2:
            continue
        out = model(**enc, labels=enc["input_ids"])
        losses.append(out.loss.item())
    mean_loss = sum(losses) / len(losses)
    return math.exp(mean_loss)


def count_trainable_params(model):
    # Note: PeftModel.from_pretrained() defaults to is_trainable=False (inference
    # mode), which sets requires_grad=False on adapter weights too. Counting by
    # parameter name instead gives the structural LoRA size regardless of mode.
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    total = sum(p.numel() for p in model.parameters())
    return lora_params, total


def main(cfg_path, adapter_dir, eval_split, n_eval_samples):
    cfg = load_config(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_ds = load_dataset(cfg["dataset_name"], split=eval_split)
    texts = eval_ds[cfg["text_field"]][:n_eval_samples]

    print("[1/2] Evaluating base model...")
    base_model = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).to(device)
    base_ppl = compute_perplexity(base_model, tokenizer, texts, device)
    print(f"  base perplexity: {base_ppl:.3f}")
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    print("[2/2] Evaluating LoRA fine-tuned model...")
    ft_base = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).to(device)
    ft_model = PeftModel.from_pretrained(ft_base, adapter_dir).to(device)
    ft_ppl = compute_perplexity(ft_model, tokenizer, texts, device)
    ft_trainable, ft_total = count_trainable_params(ft_model)
    print(f"  fine-tuned perplexity: {ft_ppl:.3f}")

    print("\n===== Summary =====")
    print(f"Base model perplexity:       {base_ppl:.3f}")
    print(f"Fine-tuned model perplexity: {ft_ppl:.3f}")
    print(f"Perplexity delta:            {ft_ppl - base_ppl:+.3f}")
    print(
        f"Trainable params (LoRA):     {ft_trainable:,} / {ft_total:,} "
        f"({100 * ft_trainable / ft_total:.3f}%)"
    )

    try:
        import lm_eval  # noqa: F401

        print(
            "\n[optional] lm-evaluation-harness is installed. See this module's "
            "README for instructions on running standard benchmark tasks "
            "(e.g., hellaswag, arc_easy) against this checkpoint."
        )
    except ImportError:
        print(
            "\n[note] Install `lm-eval` to additionally run standard "
            "lm-evaluation-harness benchmarks on this checkpoint."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--eval_split", type=str, default="train[2000:2200]")
    parser.add_argument("--n_eval_samples", type=int, default=50)
    args = parser.parse_args()
    main(args.config, args.adapter_dir, args.eval_split, args.n_eval_samples)
