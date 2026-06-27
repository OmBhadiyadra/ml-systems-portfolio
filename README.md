# ML Systems Portfolio

Hands-on projects covering the ML systems engineering stack: distributed training,
LLM inference optimization, and production model serving.

> **Status**: Actively being built and benchmarked. Results below are updated as
> each experiment is run (see individual project folders for logs and detailed numbers).

## Projects

| # | Project | Focus | Status |
|---|---------|-------|--------|
| 1 | [LoRA Fine-Tuning](./01_lora_finetuning) | PEFT/LoRA fine-tuning with PyTorch + HuggingFace | 🔧 In progress |
| 2 | [Distributed Training](./02_distributed_training) | Multi-GPU training with DDP/FSDP, CUDA/NCCL debugging | 🔧 In progress |
| 3 | [Quantization & Speculative Decoding](./03_quantization_speculative_decoding) | 4-bit/8-bit quantization, KV cache, speculative decoding | 🔧 In progress |
| 4 | [Production Serving](./04_fastapi_serving) | FastAPI + Docker inference endpoint with latency profiling | 🔧 In progress |

## Stack

PyTorch · HuggingFace Transformers · PEFT (LoRA) · Accelerate · Torch Distributed (DDP/FSDP) ·
bitsandbytes/GPTQ · FastAPI · Docker

## About

Built by [Om Bhadiyadra](https://linkedin.com/in/ombhadiyadra) — MS Computer Science,
UMass Dartmouth. Each project folder contains its own README with setup instructions,
what was implemented, and benchmark results from real runs.
