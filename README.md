# LLM Systems Engineering: Training, Optimization & Production Serving

A hands-on systems engineering portfolio covering the full lifecycle of working with
large language models — fine-tuning, distributed training, inference optimization,
and production deployment.

## Why this repo exists

Most ML portfolios stop at "I trained a model." This one is about what happens
*around* training: making it run efficiently across multiple GPUs, making inference
fast and memory-efficient, and packaging a model as something a real product could
actually call.

## Structure

| Folder | Focus | Status |
|---|---|---|
| [`01_lora_finetuning`](./01_lora_finetuning) | PEFT/LoRA fine-tuning with a custom PyTorch training loop, evaluated against the base model | In progress |
| [`02_distributed_training`](./02_distributed_training) | Multi-GPU training with DDP & FSDP, CUDA/NCCL debugging | Coming soon |
| [`03_inference_optimization`](./03_inference_optimization) | Post-training quantization, KV cache, speculative decoding | Coming soon |
| [`04_production_serving`](./04_production_serving) | FastAPI inference endpoint, Dockerized, load-tested | Coming soon |

## Tech stack

PyTorch · HuggingFace Transformers · PEFT · Accelerate · Torch Distributed (DDP/FSDP)
· bitsandbytes / GPTQ · lm-evaluation-harness · FastAPI · Docker

## Setup

```bash
git clone https://github.com/OmBhadiyadra/llm-systems-engineering.git
cd llm-systems-engineering
pip install -r requirements.txt
```

Each subfolder has its own README with exact run instructions. Training/eval scripts
are designed to run on a free-tier Google Colab GPU (T4) — no paid infrastructure
required to reproduce the results.

Model weights and checkpoints are not committed to this repo (see `.gitignore`) —
only code, configs, and result tables are tracked.

## Author

Om Bhadiyadra — [LinkedIn](https://www.linkedin.com/in/ombhadiyadra)
