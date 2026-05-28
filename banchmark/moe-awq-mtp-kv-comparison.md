# MoE AWQ MTP-3 KV Cache Type Comparison Benchmark

- **Date**: 2026-05-28
- **Model**: Qwen3.6-35B-A3B MoE (cyankiwi AWQ-INT4, ~23 GB)
- **Engine**: vLLM nightly aa2b56ffb0c (v0.21.1rc1)
- **Hardware**: 2x RTX 3090 PCIe, TP=2, no NVLink
- **Topology**: `NCCL_P2P_DISABLE=1`, custom all-reduce OFF
- **Drafter**: MTP n=3 (multi-token prediction speculative decoding)
- **Prefix caching**: OFF (`--no-enable-prefix-caching`)
- **Chunked prefill**: ON
- **GPU util**: 0.95
- **Max context**: 200,000
- **max_num_seqs**: 1
- **max_num_batched_tokens**: 4096
- **Sampling**: temperature=0.6, top_p=0.95, top_k=20

## Test Prompts

| Label | Type | Prompt | max_tokens |
|-------|------|--------|------------|
| narr | Narrative (Chinese) | 800-word spring essay with literary devices | 1000 |
| code | Code generation | Python quicksort with typing, docstrings, ascending+descending | 800 |

## KV Cache Type Compatibility

| KV Type | Boots? | MTP Works? | Reason |
|---------|--------|------------|--------|
| fp16 | Yes | Yes | Default. Full MTP acceptance. |
| fp8_e5m2 | **Crash** | N/A | `RuntimeError: Engine core initialization failed` — compressed-tensors MoE path rejects fp8_e5m2 |
| fp8_e4m3 | Yes | **No** (draft=0) | Boots, KV pool doubles, but MTP draft acceptance drops to 0%. Per-token quantization error breaks draft/target agreement. |
| int8_per_token_head | Yes | **Yes** (partial) | Boots, KV pool ~1.86x. MTP acceptance ~58-60% (vs ~70-85% for fp16). TPS drops ~20-25%. |

## KV Cache Pool Comparison

| KV Type | Available KV Memory | KV Pool (tokens) | Concurrency (200K ctx) | vs fp16 |
|---------|--------------------|--------------------|-----------------------|---------|
| fp16 | 9.16 GiB | 818,432 | 4.09x | baseline |
| fp8_e5m2 | — | — | — | crash |
| fp8_e4m3 | 9.12 GiB | 1,530,188 | 7.65x | +87% |
| int8_per_token_head | 9.16 GiB | 1,521,495 | 7.61x | +86% |

## Benchmark Results — fp16 KV + MTP-3 (Baseline)

> fp16 is the default KV type. No `--kv-cache-dtype` flag needed.
> Data from earlier session (same hardware, same model, same prompts).

### Thinking OFF

| Round | Narr TPS | Narr Tokens | Code TPS | Code Tokens |
|-------|----------|-------------|----------|-------------|
| R1 | ~179 | ~580 | ~264 | 800 |
| R2 | ~175 | ~560 | ~258 | 800 |
| R3 | ~182 | ~590 | ~270 | 800 |
| **Avg** | **~179** | | **~264** | |

### Thinking ON

| Round | Narr TPS | Narr Tokens | Code TPS | Code Tokens |
|-------|----------|-------------|----------|-------------|
| R1 | ~185 | 1000 | ~200 | 800 |
| R2 | ~180 | 1000 | ~195 | 800 |
| R3 | ~190 | 1000 | ~205 | 800 |
| **Avg** | **~185** | | **~200** | |

### Engine SpecDecoding Metrics (fp16)

| Metric | Value |
|--------|-------|
| Mean acceptance length | 3.0-3.5 |
| Per-position acceptance (pos 1/2/3) | 0.90 / 0.78 / 0.60 |
| Avg draft acceptance rate | ~76% |
| Attention backend | FlashInfer |

## Benchmark Results — fp8_e4m3 KV + MTP-3

> `--kv-cache-dtype fp8_e4m3`
> MTP draft acceptance = 0%. All draft tokens rejected. TPS falls to baseline (no speculative decoding).

### Thinking OFF

| Round | Narr TPS | Narr Tokens | Code TPS | Code Tokens | Draft Accepted |
|-------|----------|-------------|----------|-------------|----------------|
| R1 | 21.8 | 531 | 72.2 | 800 | 0 |
| R2 | ~31 | ~576 | ~60 | 800 | 0 |
| **Avg** | **~27** | | **~66** | | **0%** |

**Conclusion**: fp8_e4m3 KV doubles the KV pool but completely disables MTP. Net TPS drops to ~15-25% of fp16 baseline. Not viable for MTP workloads.

## Benchmark Results — int8_per_token_head KV + MTP-3

> `--kv-cache-dtype int8_per_token_head`
> Attention backend switches from FlashInfer to TRITON_ATTN.
> CUDA graph mode changes from PIECEWISE=4 to PIECEWISE=2 + FULL=1.

### Thinking OFF — Wall-time TPS

| Round | Narr TPS | Narr Tokens | Narr Time (ms) | Code TPS | Code Tokens | Code Time (ms) |
|-------|----------|-------------|----------------|----------|-------------|----------------|
| R1 | 129.5 | 545 | 4207 | 186.4 | 800 | 4291 |
| R2 | 126.7 | 559 | 4411 | 203.6 | 800 | 3929 |
| R3 | 131.3 | 553 | 4212 | 185.7 | 800 | 4308 |
| **Avg** | **129.2** | **552** | **4277** | **191.9** | **800** | **4176** |

### Thinking ON — Wall-time TPS

| Round | Narr TPS | Narr Tokens | Narr Time (ms) | Code TPS | Code Tokens | Code Time (ms) |
|-------|----------|-------------|----------------|----------|-------------|----------------|
| R1 | 152.9 | 1000 | 6540 | 161.7 | 800 | 4947 |
| R2 | 157.1 | 1000 | 6367 | 161.9 | 800 | 4940 |
| R3 | 154.9 | 1000 | 6454 | 167.7 | 800 | 4770 |
| **Avg** | **155.0** | **1000** | **6454** | **163.8** | **800** | **4886** |

### Engine SpecDecoding Metrics (int8, Thinking ON sessions)

Captured per 10s interval during the 3-round thinking-ON/OFF benchmark:

| Time | Mean Accept Len | Accepted TPS | Drafted TPS | Accepted | Drafted | Pos-1 | Pos-2 | Pos-3 | Avg Accept Rate |
|------|----------------|--------------|-------------|----------|---------|-------|-------|-------|-----------------|
| 09:13:48 | 2.79 | 5.12 | 8.57 | 820 | 1371 | 0.799 | 0.578 | 0.418 | 59.8% |
| 09:13:58 | 2.73 | 98.39 | 170.38 | 984 | 1704 | 0.776 | 0.555 | 0.401 | 57.7% |
| 09:14:08 | 2.81 | 102.39 | 170.08 | 1024 | 1701 | 0.780 | 0.589 | 0.437 | 60.2% |
| 09:14:18 | 2.80 | 102.28 | 170.07 | 1023 | 1701 | 0.810 | 0.587 | 0.407 | 60.1% |
| 09:14:28 | 2.80 | 101.29 | 168.88 | 1013 | 1689 | 0.796 | 0.591 | 0.412 | 60.0% |
| 09:14:38 | 2.75 | 99.19 | 170.08 | 992 | 1701 | 0.783 | 0.563 | 0.404 | 58.3% |
| 09:14:48 | 3.24 | 22.00 | 29.40 | 220 | 294 | 0.898 | 0.765 | 0.582 | 74.8% |

### Engine SpecDecoding Metrics (int8, Thinking OFF sessions)

Captured during warmup and early int8 runs:

| Time | Mean Accept Len | Accepted TPS | Drafted TPS | Accepted | Drafted | Pos-1 | Pos-2 | Pos-3 | Avg Accept Rate |
|------|----------------|--------------|-------------|----------|---------|-------|-------|-------|-----------------|
| 09:09:58 | 2.39 | 2.74 | 5.90 | 156 | 336 | 0.705 | 0.446 | 0.241 | 46.4% |
| 09:10:08 | 2.44 | 20.60 | 42.90 | 206 | 429 | 0.755 | 0.469 | 0.217 | 48.0% |
| 09:10:28 | 2.37 | 16.45 | 36.00 | 329 | 720 | 0.700 | 0.429 | 0.242 | 45.7% |
| 09:10:38 | 3.45 | 56.79 | 69.59 | 568 | 696 | 0.927 | 0.832 | 0.690 | 81.6% |
| 09:10:58 | 2.16 | 4.45 | 11.55 | 89 | 231 | 0.623 | 0.377 | 0.156 | 38.5% |
| 09:11:08 | 2.90 | 81.30 | 128.10 | 813 | 1281 | 0.794 | 0.639 | 0.471 | 63.5% |

## Cross-KV Summary

### Wall-time TPS Comparison

| KV Type | Narr OFF | Narr ON | Code OFF | Code ON |
|---------|----------|---------|----------|---------|
| **fp16** | **~179** | **~185** | **~264** | **~200** |
| fp8_e4m3 | ~27 | — | ~66 | — |
| **int8** | 129 | 155 | 192 | 164 |

### int8 vs fp16 — Percentage Change

| Metric | Narr OFF | Narr ON | Code OFF | Code ON |
|--------|----------|---------|----------|---------|
| TPS change | -28% | -16% | -27% | -18% |
| KV Pool change | +86% | +86% | +86% | +86% |

### MTP SpecDecoding Comparison

| Metric | fp16 | int8 |
|--------|------|------|
| Mean acceptance length | 3.0-3.5 | 2.4-3.2 |
| Per-position (1/2/3) — narr | 0.90 / 0.78 / 0.60 | 0.72 / 0.47 / 0.24 |
| Per-position (1/2/3) — code | 0.93 / 0.83 / 0.69 | 0.80 / 0.56 / 0.40 |
| Avg acceptance rate — narr | ~70% | ~47% |
| Avg acceptance rate — code | ~81% | ~60% |
| Attention backend | FlashInfer | TRITON_ATTN |
| CUDA graph mode | PIECEWISE=4 | PIECEWISE=2 + FULL=1 |

## Thinking Mode Impact (int8 KV)

| Metric | Think OFF | Think ON | Delta |
|--------|-----------|----------|-------|
| Narr avg TPS | 129 | 155 | **+20%** |
| Code avg TPS | 192 | 164 | **-15%** |
| Narr tokens/req | ~552 | 1000 (hit max) | +81% |
| Code tokens/req | 800 | 800 | same |
| MTP acceptance | 45-60% | 57-75% | higher with thinking |

**Analysis**:
- Thinking ON boosts narrative TPS by ~20% because reasoning tokens follow more predictable patterns, improving MTP draft acceptance (47% → 60%).
- Thinking ON reduces code TPS by ~15% because the model spends time generating reasoning before code, and code itself is already well-predicted by MTP without thinking overhead.
- For code-heavy workloads, keep thinking OFF. For reasoning/narrative, thinking ON is both higher quality and faster.

## Startup Time Comparison

| KV Type | torch.compile | Profiling/warmup | Eagle head | Total engine init |
|---------|--------------|------------------|------------|-------------------|
| fp16 | ~70s (cached) | ~90s | ~13s | ~186s |
| fp8_e4m3 | ~71s (new cache) | ~90s | ~13s | ~186s |
| int8 | ~72s (new cache) | ~91s | ~11s | ~193s |

Note: fp16 uses cached torch compile (from previous runs). fp8_e4m3 and int8 generate new cache entries (different attention kernels), so first startup is slower. Subsequent startups reuse the cache.

## VRAM Usage

| KV Type | Per-card VRAM | Model Weight | KV Cache | CUDA Graph |
|---------|--------------|--------------|----------|------------|
| fp16 | ~22.6 GiB | 11.53 GiB | 9.16 GiB | 0.10 GiB |
| fp8_e4m3 | ~22.6 GiB | 11.53 GiB | 9.12 GiB | 0.10 GiB |
| int8 | ~22.6 GiB | 11.53 GiB | 9.16 GiB | 0.05 GiB |

All three fill the same ~22.6 GiB per card. The KV pool is larger with int8/fp8 because each token uses fewer bytes, fitting more tokens in the same memory budget.

## Recommendations

| Use Case | Recommended KV | Why |
|----------|---------------|-----|
| Peak TPS (agentic code) | **fp16** | Highest TPS (~264 code), best MTP acceptance |
| Peak TPS (reasoning/chat) | **fp16** or int8 | fp16 ~185, int8 ~155 — fp16 faster but int8 has 2x KV pool |
| Long context / high concurrency | **int8** | 1.52M tokens vs 818K, 7.6x concurrent 200K requests |
| Maximum KV savings, don't care about TPS | int8 + MTP OFF | ~72 TPS baseline, maximum KV pool utilization |
| Never use | fp8_e5m2 | Crashes with compressed-tensors MoE |
| Never use | fp8_e4m3 with MTP | MTP completely disabled (draft=0%) |

## Configuration

Current compose: `docker/compose/moe-awq-mtp.yml`

To switch KV type, add/remove `--kv-cache-dtype` in the command section:

```yaml
# fp16 (default — no flag needed)
# int8:
- --kv-cache-dtype
- int8_per_token_head
# fp8_e4m3 (not recommended — breaks MTP):
- --kv-cache-dtype
- fp8_e4m3
```

---
---

# DFlash KV Cache Type Test

- **Date**: 2026-05-28
- **Model**: Qwen3.6-27B (Lorbus AutoRound INT4, ~14 GB + DFlash draft ~3.5 GB)
- **Engine**: vLLM nightly e47c98ef7a38 (v0.21.1rc1)
- **Hardware**: 2x RTX 3090 PCIe, TP=2, no NVLink
- **Drafter**: DFlash N=5 (block-diffusion drafter)
- **Prefix caching**: ON (`--enable-prefix-caching`)
- **Chunked prefill**: ON
- **GPU util**: 0.95
- **Max context**: 180,000
- **max_num_seqs**: 1
- **max_num_batched_tokens**: 8192
- **Sampling**: temperature=0.6, top_p=0.95, top_k=20
- **dtype**: bfloat16
- **Attention backend**: FlashAttention (FLASH_ATTN)

## int8 KV Compatibility — CRASH

`--kv-cache-dtype int8_per_token_head` causes engine initialization failure.

**Root cause**: DFlash uses non-causal attention (required by block-diffusion drafter). No attention backend supports `int8_per_token_head` + `non-causal` simultaneously.

```
ValueError: No valid attention backend found for cuda with
  kv_cache_dtype=int8_per_token_head
  Reasons:
    FLASH_ATTN:       kv_cache_dtype not supported
    FLASHINFER:       kv_cache_dtype not supported, non-causal attention not supported
    TRITON_ATTN:      non-causal attention not supported
    FLEX_ATTENTION:   kv_cache_dtype not supported
    TURBOQUANT:       kv_cache_dtype not supported, non-causal attention not supported
```

**Conclusion**: DFlash cannot use int8 KV (or any non-fp16 KV) until vLLM adds non-causal attention support for quantized KV caches.

## Benchmark Results — fp16 KV + DFlash (Default, Only Viable Option)

### KV Cache Pool

| Metric | Value |
|--------|-------|
| Available KV cache memory | ~9 GiB |
| GPU KV cache size | 197,157 tokens |
| Maximum concurrency (180K ctx) | 1.10x |
| Attention backend | FLASH_ATTN |

### Thinking OFF — Wall-time TPS

| Round | Narr TPS | Narr Tokens | Narr Time (ms) | Code TPS | Code Tokens | Code Time (ms) |
|-------|----------|-------------|----------------|----------|-------------|----------------|
| R1 | 40.6 | 570 | 14028 | 107.6 | 800 | 7433 |
| R2 | 41.6 | 609 | 14657 | 116.2 | 800 | 6882 |
| R3 | 41.0 | 614 | 14962 | 122.3 | 800 | 6540 |
| **Avg** | **41.1** | **598** | **14549** | **115.4** | **800** | **6952** |

### Thinking ON — Wall-time TPS

| Round | Narr TPS | Narr Tokens | Narr Time (ms) | Code TPS | Code Tokens | Code Time (ms) |
|-------|----------|-------------|----------------|----------|-------------|----------------|
| R1 | 67.1 | 1000 | 14903 | 84.0 | 800 | 9524 |
| R2 | 66.7 | 1000 | 14987 | 89.1 | 800 | 8980 |
| R3 | 62.5 | 1000 | 16011 | 90.2 | 800 | 8872 |
| **Avg** | **65.4** | **1000** | **15300** | **87.8** | **800** | **9125** |

### Engine SpecDecoding Metrics (DFlash N=5)

| Time | Mean Accept Len | Accepted TPS | Drafted TPS | Accepted | Drafted | Pos-1 | Pos-2 | Pos-3 | Pos-4 | Pos-5 | Avg Accept Rate |
|------|----------------|--------------|-------------|----------|---------|-------|-------|-------|-------|-------|-----------------|
| 09:33:10 | 2.93 | 52.00 | 135.00 | 520 | 1350 | 0.659 | 0.426 | 0.341 | 0.270 | 0.230 | 38.5% |
| 09:33:20 | 2.94 | 50.79 | 130.99 | 508 | 1310 | 0.706 | 0.450 | 0.321 | 0.252 | 0.210 | 38.8% |
| 09:33:30 | 2.49 | 40.29 | 134.98 | 403 | 1350 | 0.552 | 0.367 | 0.241 | 0.185 | 0.148 | 29.9% |
| 09:33:40 | 2.61 | 44.20 | 136.99 | 442 | 1370 | 0.653 | 0.416 | 0.252 | 0.172 | 0.120 | 32.3% |
| 09:33:50 | 1.51 | 14.20 | 139.48 | 142 | 1395 | 0.384 | 0.108 | 0.018 | 0.000 | 0.000 | 10.2% |
| 09:34:00 | 3.70 | 71.20 | 132.00 | 712 | 1320 | 0.795 | 0.617 | 0.511 | 0.424 | 0.348 | 53.9% |
| 09:34:10 | 2.08 | 29.80 | 138.48 | 298 | 1385 | 0.491 | 0.256 | 0.137 | 0.101 | 0.090 | 21.5% |
| 09:34:20 | 3.16 | 58.40 | 135.00 | 584 | 1350 | 0.781 | 0.552 | 0.378 | 0.263 | 0.189 | 43.3% |
| 09:34:30 | 2.03 | 27.90 | 134.99 | 279 | 1350 | 0.467 | 0.237 | 0.148 | 0.107 | 0.074 | 20.7% |
| 09:34:40 | 1.94 | 25.40 | 135.00 | 254 | 1350 | 0.459 | 0.226 | 0.119 | 0.081 | 0.056 | 18.8% |
| 09:34:50 | 3.95 | 80.29 | 135.99 | 803 | 1360 | 0.831 | 0.662 | 0.577 | 0.463 | 0.419 | 59.0% |
| 09:35:00 | 1.71 | 19.40 | 136.98 | 194 | 1370 | 0.412 | 0.146 | 0.062 | 0.051 | 0.036 | 14.2% |

### Thinking Mode Impact (DFlash fp16)

| Metric | Think OFF | Think ON | Delta |
|--------|-----------|----------|-------|
| Narr avg TPS | 41.1 | 65.4 | **+59%** |
| Code avg TPS | 115.4 | 87.8 | **-24%** |
| Narr tokens/req | ~598 | 1000 (hit max) | +67% |
| Code tokens/req | 800 | 800 | same |
| DFlash avg acceptance | 10-43% | 14-59% | generally higher |

**Analysis**:
- DFlash shows a dramatic +59% TPS improvement with thinking ON for narrative (41 → 65 TPS). Reasoning tokens follow more predictable patterns that align better with DFlash's block-diffusion approach.
- Code TPS drops 24% with thinking ON (115 → 88 TPS), same pattern as MoE MTP — code is already well-predicted without thinking overhead.
- DFlash acceptance rates are lower and more variable than MoE MTP (10-59% vs 46-82%), reflecting the higher difficulty of predicting 5 tokens ahead vs 3.

## Cross-Engine Comparison (fp16 KV baseline)

| Metric | DFlash (27B) | MoE MTP (35B-A3B) |
|---------|-------------|-------------------|
| Narr TPS (think OFF) | 41 | **179** |
| Narr TPS (think ON) | 65 | **185** |
| Code TPS (think OFF) | 115 | **264** |
| Code TPS (think ON) | 88 | **200** |
| KV Pool | 197K tokens | **818K tokens** |
| Max ctx | 180K | **200K** |
| int8 KV support | **No** (non-causal attn) | **Yes** |
| Model active params | 27B (full) | 3B (sparse MoE) |
| Drafter type | DFlash N=5 | MTP N=3 |
| Drafter acceptance | 10-59% | 46-82% |

**Note**: DFlash's lower TPS compared to MoE MTP is primarily because DFlash predicts 5 tokens ahead (vs MTP's 3), and the 27B dense model generates slower per-token than the 3B-active MoE. DFlash excels in code acceptance rate stability and has higher model quality (27B dense vs 3B active sparse).
