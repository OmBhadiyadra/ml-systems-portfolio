# 02 — Distributed Transformer Training (DDP / FSDP)

**Status: coming in the next step of this project.**

This module will cover:

- Multi-GPU training using PyTorch `DistributedDataParallel` (DDP) and
  `FullyShardedDataParallel` (FSDP)
- Gradient checkpointing and parameter sharding to reduce per-GPU memory
- Debugging real CUDA out-of-memory and NCCL synchronization failures
- Benchmarking training throughput and memory footprint: single-GPU vs.
  distributed configurations
