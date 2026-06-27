# 02 — Distributed Transformer Training (DDP / FSDP)

Trains the same causal LM from Module 1, but across **multiple GPUs**, using both
PyTorch `DistributedDataParallel` (DDP) and `FullyShardedDataParallel` (FSDP).
Benchmarks throughput and peak per-GPU memory across single-GPU, DDP, and FSDP
configurations.

## Why this needs Kaggle, not Colab

DDP and FSDP only mean something with more than one GPU. **Colab's free tier gives
you exactly one GPU** — so distributed code would "work" trivially but wouldn't
demonstrate anything. **Kaggle Notebooks' free tier gives 2x T4 GPUs**, which is
what this module actually needs.

## What's here

| File | Purpose |
|---|---|
| `config.yaml` | Model, data, and training hyperparameters |
| `model_utils.py` | Shared model/tokenizer/dataloader construction |
| `train_distributed.py` | Single script, run 3 ways: plain `python` (1 GPU), `torchrun --strategy ddp`, `torchrun --strategy fsdp` |
| `benchmark_distributed.py` | Runs all 3 configurations and prints a comparison table |

## Running on Kaggle (free, 2x T4 GPUs)

1. Go to [kaggle.com](https://kaggle.com) → sign up / log in
2. Click **Create → New Notebook**
3. Right sidebar → **Settings → Accelerator → GPU T4 x2**
4. Right sidebar → **Settings → Internet → On** (off by default; needed to clone the repo and download the model)
5. In the first cell:

```python
!git clone https://github.com/OmBhadiyadra/ml-systems-portfolio.git /kaggle/working/ml-systems-portfolio
!pip install -q -U transformers peft accelerate datasets pyyaml
!pip uninstall -y torchao -q
```

6. Single-GPU baseline:

```python
!python /kaggle/working/ml-systems-portfolio/02_distributed_training/train_distributed.py \
  --config /kaggle/working/ml-systems-portfolio/02_distributed_training/config.yaml \
  --strategy ddp
```

7. DDP across both GPUs:

```python
!torchrun --nproc_per_node=2 /kaggle/working/ml-systems-portfolio/02_distributed_training/train_distributed.py \
  --config /kaggle/working/ml-systems-portfolio/02_distributed_training/config.yaml \
  --strategy ddp
```

8. FSDP across both GPUs:

```python
!torchrun --nproc_per_node=2 /kaggle/working/ml-systems-portfolio/02_distributed_training/train_distributed.py \
  --config /kaggle/working/ml-systems-portfolio/02_distributed_training/config.yaml \
  --strategy fsdp
```

Each run prints a `DONE` summary line with total time, throughput (samples/sec),
and peak per-GPU memory. Copy those into the **Results** table below.

## Results

_To be filled in after running on Kaggle._

| Configuration | World size | Throughput (samples/s) | Peak mem/GPU (MB) |
|---|---|---|---|
| Single-GPU baseline | 1 | TBD | TBD |
| DDP | 2 | TBD | TBD |
| FSDP | 2 | TBD | TBD |

## Troubleshooting log

_Real issues hit while running this on Kaggle will be documented here as they
come up — same approach as Module 1, where a `torchao`/`peft` version conflict
and a `datasets`/`torchvision` import bug were hit and fixed live._

## Notes

- `train_distributed.py` is one script with a `--strategy` flag rather than two
  separate DDP/FSDP scripts, since the training loop is identical — only the
  model-wrapping step differs.
- FSDP wraps each transformer decoder layer as its own unit
  (`transformer_auto_wrap_policy`), so parameters/gradients/optimizer state are
  sharded layer-by-layer across GPUs rather than fully replicated like DDP.
- At this model size (0.5B params), FSDP's memory advantage over DDP may be
  modest — the benefit grows with model size, since FSDP shards optimizer state
  that DDP must fully replicate on every GPU. Actual numbers from the run are
  reported above, whatever they turn out to be.
