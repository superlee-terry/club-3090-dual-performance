# vLLM PR #40886 overlay — AWQ compressed-tensors MoE key remapping

## What this fixes

`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` ships in `compressed-tensors` pack-quantized
format, which stores per-expert weights with `_packed` (int32) and `_scale` (bfloat16)
suffixes. The vLLM `gemma4.py::_weight_iterator` (as of bf610c2f, 2026-05-15) only
handles the float-checkpoint key pattern and does not remap the `_packed` / `_scale`
suffixed variants on MoE expert weights. Without this overlay, model load fails with
a `KeyError` on the first packed expert key.

The fix is from [vLLM PR #40886](https://github.com/vllm-project/vllm/pull/40886) by
@tajwali. PR author tested on **RTX 3090 24 GB** (same SKU as ours) with vLLM 0.19.1.
The patch is +23 / -0 — pure insertion of 4 conditional branches in
`_weight_iterator` that intercept the four `_packed` / `_scale` MoE key patterns
and yield per-expert float-shaped keys that the existing FusedMoE loader path expects.

## When to drop this overlay

When **both** of these are true:
1. PR #40886 has merged upstream
2. The engine's pinned nightly SHA is past the merge commit

## Source PR

- PR head: `tajwali/vllm` @ `652819dad0bf9bbb0436d6660822e7aff30c3ff0`
- Vendored as of 2026-05-15
