# 03 — LLM Inference Optimization

**Status: coming in the next step of this project.**

This module will cover:

- Post-training 4-bit / 8-bit quantization (bitsandbytes / GPTQ), measuring
  memory footprint reduction vs. the unquantized baseline
- KV cache mechanics and their effect on generation latency
- Speculative decoding with a draft-model approach, measuring inference
  throughput improvement over standard autoregressive generation
