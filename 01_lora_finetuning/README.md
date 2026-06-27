# 01 — LoRA Fine-Tuning (PyTorch + HuggingFace + PEFT)

Fine-tunes a pretrained Transformer using LoRA (Low-Rank Adaptation) via a **custom
PyTorch training loop** — manual optimizer/scheduler stepping, gradient accumulation,
and periodic checkpointing (not the HuggingFace `Trainer` API, by design, to
demonstrate the underlying mechanics).

The fine-tuned model is then evaluated against the base model on held-out
perplexity, alongside the LoRA trainable-parameter reduction.

## What's here

| File | Purpose |
|---|---|
| `config.yaml` | Model, LoRA, and training hyperparameters |
| `train_lora.py` | Custom training loop, LoRA adapter training |
| `evaluate.py` | Compares base vs. fine-tuned perplexity, reports trainable param % |

## Model & data

- **Base model:** `Qwen/Qwen2.5-0.5B` — small enough to fine-tune on a free Colab T4 GPU
- **Dataset:** `tatsu-lab/alpaca` (instruction-following), first 2,000 examples for training

Both are swappable in `config.yaml`.

## Running on Google Colab (free GPU)

1. Open [colab.research.google.com](https://colab.research.google.com) → New Notebook
2. **Runtime → Change runtime type → T4 GPU**
3. Run these cells in order:

```python
!git clone https://github.com/OmBhadiyadra/llm-systems-engineering.git
%cd llm-systems-engineering
!pip install -q -r requirements.txt
```

```python
%cd 01_lora_finetuning
!python train_lora.py --config config.yaml
```

```python
!python evaluate.py --config config.yaml --adapter_dir outputs/lora_run/final_adapter
```

4. Copy the printed metrics from the `evaluate.py` output into the **Results** table
   below and commit the update.

## Results

Run on a Colab T4 GPU, 1 epoch, 2,000 examples from `tatsu-lab/alpaca`.

| Model | Perplexity | Trainable Params | % of Total |
|---|---|---|---|
| Base (Qwen2.5-0.5B) | 7.326 | — | — |
| LoRA fine-tuned | 3.307 | 540,672 | 0.109% |

Fine-tuning reduced perplexity by 4.019 (~55%) while updating only 0.109% of
total model parameters — demonstrating LoRA's core value: near-full-model
adaptation quality at a fraction of the trainable parameter cost.

## Notes

- LoRA adapters (not full model weights) are the only trainable parameters —
  `model.print_trainable_parameters()` during training shows the exact reduction.
- Checkpoints are written to `outputs/` locally and are git-ignored (too large to
  commit); only code, configs, and results are version-controlled.
