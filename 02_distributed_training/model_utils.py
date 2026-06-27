"""
Shared model and dataloader construction for distributed training scripts.
"""

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def build_dataloader(cfg, tokenizer, rank, world_size):
    from datasets import load_dataset

    raw = load_dataset(cfg["dataset_name"], split=cfg["dataset_split"])

    def tokenize_fn(example):
        return tokenizer(
            example[cfg["text_field"]],
            truncation=True,
            max_length=cfg["max_seq_length"],
            padding="max_length",
        )

    tokenized = raw.map(tokenize_fn, batched=False, remove_columns=raw.column_names)
    tokenized.set_format(type="torch")

    sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            tokenized, num_replicas=world_size, rank=rank, shuffle=True
        )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(
        tokenized,
        batch_size=cfg["per_device_batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collator,
    )
    return loader
