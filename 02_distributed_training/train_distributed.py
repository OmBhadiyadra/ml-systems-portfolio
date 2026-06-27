"""
Distributed Transformer Training (DDP / FSDP)

Works in three modes depending on how it's launched:
  1. Plain `python train_distributed.py ...`        -> single-GPU baseline (world_size=1)
  2. `torchrun --nproc_per_node=2 ... --strategy ddp`  -> DistributedDataParallel across 2 GPUs
  3. `torchrun --nproc_per_node=2 ... --strategy fsdp` -> FullyShardedDataParallel across 2 GPUs

Logs per-step throughput (samples/sec) and peak per-GPU memory, and prints a
final summary line consumed by benchmark_distributed.py.
"""

import argparse
import functools
import os
import time

import torch
import torch.distributed as dist
import yaml
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP

from model_utils import build_dataloader, build_model_and_tokenizer


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def wrap_model(model, strategy, local_rank, world_size):
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if world_size <= 1:
        return model, device

    if strategy == "ddp":
        model = DDP(model, device_ids=[local_rank])
    elif strategy == "fsdp":
        # Wrap each transformer decoder layer as its own FSDP unit so that
        # parameters, gradients, and optimizer state are sharded layer-by-layer
        # across GPUs, rather than fully replicated like DDP.
        wrap_policy = None
        try:
            decoder_layer_cls = type(model.model.layers[0])
            wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={decoder_layer_cls},
            )
        except AttributeError:
            pass  # fall back to wrapping the whole model as one FSDP unit

        model = FSDP(model, auto_wrap_policy=wrap_policy, device_id=local_rank)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return model, device


def main(cfg_path, strategy):
    cfg = load_config(cfg_path)
    rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0

    if is_main:
        print(
            f"[setup] strategy={strategy} world_size={world_size} "
            f"cuda_available={torch.cuda.is_available()}"
        )

    base_model, tokenizer = build_model_and_tokenizer(cfg["model_name"])
    model, device = wrap_model(base_model, strategy, local_rank, world_size)
    loader = build_dataloader(cfg, tokenizer, rank, world_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    model.train()
    t0 = time.time()
    peak_mem_mb = 0.0
    global_step = 0
    max_steps = cfg.get("max_steps", float("inf"))

    for epoch in range(cfg["num_epochs"]):
        if world_size > 1 and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if torch.cuda.is_available():
                peak_mem_mb = max(
                    peak_mem_mb,
                    torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                )

            if is_main and global_step % cfg["log_every"] == 0:
                elapsed = time.time() - t0
                throughput = (
                    global_step * cfg["per_device_batch_size"] * world_size / elapsed
                )
                print(
                    f"[{strategy}] step {global_step} loss={loss.item():.4f} "
                    f"elapsed={elapsed:.1f}s throughput={throughput:.2f} samples/s "
                    f"peak_mem={peak_mem_mb:.0f}MB"
                )

            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    total_time = time.time() - t0
    if is_main:
        total_samples = global_step * cfg["per_device_batch_size"] * world_size
        print(
            f"\n[{strategy}] DONE world_size={world_size} "
            f"total_steps={global_step} total_time={total_time:.1f}s "
            f"throughput={total_samples / total_time:.2f} samples/s "
            f"peak_mem_per_gpu={peak_mem_mb:.0f}MB"
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--strategy", type=str, choices=["ddp", "fsdp"], default="ddp")
    args = parser.parse_args()
    main(args.config, args.strategy)
