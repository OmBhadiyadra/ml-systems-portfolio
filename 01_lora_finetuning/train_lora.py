"""
LoRA Fine-Tuning of a Pretrained Transformer (PyTorch + HuggingFace + PEFT)

Implements a custom training loop (not the HF Trainer API) with:
  - Manual optimizer + LR scheduler stepping
  - Gradient accumulation
  - Periodic checkpointing
  - Loss / throughput logging

Usage:
    python train_lora.py --config config.yaml
"""

import argparse
import math
import os
import time

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(cfg, tokenizer):
    raw = load_dataset(cfg["dataset_name"], split=cfg["dataset_split"])

    def tokenize_fn(example):
        text = example[cfg["text_field"]]
        return tokenizer(
            text,
            truncation=True,
            max_length=cfg["max_seq_length"],
            padding="max_length",
        )

    tokenized = raw.map(tokenize_fn, batched=False, remove_columns=raw.column_names)
    tokenized.set_format(type="torch")
    return tokenized


def main(cfg_path):
    cfg = load_config(cfg_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
    model.to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = build_dataset(cfg, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(
        dataset,
        batch_size=cfg["per_device_batch_size"],
        shuffle=True,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    grad_accum_steps = cfg["gradient_accumulation_steps"]
    num_update_steps_per_epoch = math.ceil(len(loader) / grad_accum_steps)
    total_update_steps = num_update_steps_per_epoch * cfg["num_epochs"]

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.get("warmup_steps", 0),
        num_training_steps=total_update_steps,
    )

    os.makedirs(cfg["output_dir"], exist_ok=True)

    model.train()
    global_step = 0
    running_loss = 0.0
    t0 = time.time()

    for epoch in range(cfg["num_epochs"]):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            running_loss += loss.item()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.get("max_grad_norm", 1.0)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg["log_every"] == 0:
                    elapsed = time.time() - t0
                    print(
                        f"[epoch {epoch}] step {global_step}/{total_update_steps} "
                        f"loss={running_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                        f"elapsed={elapsed:.1f}s"
                    )
                    running_loss = 0.0

                if global_step % cfg["checkpoint_every"] == 0:
                    ckpt_dir = os.path.join(
                        cfg["output_dir"], f"checkpoint-{global_step}"
                    )
                    model.save_pretrained(ckpt_dir)
                    print(f"[checkpoint] saved adapter weights to {ckpt_dir}")

    final_dir = os.path.join(cfg["output_dir"], "final_adapter")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[done] final LoRA adapter saved to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
