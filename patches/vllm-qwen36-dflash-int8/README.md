# vLLM Qwen3.6-27B DFlash + per-token-head INT8 KV — overlay

**Purpose:** Stack **PR #41703** (z-lab DFlash drafter) + **PR #40391** (lisp19
per-token-head fp8 KV page-size fix) + **PR #42102** (our DFlash + KV-quant
coexistence fix) — to deliver **long-context DFlash spec-decode on Qwen3.6-27B**
with INT8 PTH KV for Ampere consumer GPUs.

**Base nightly:** `e47c98ef7a38792996e452ef53914e21e41928e9` (2026-05-06) — recorded in `VERSION_PIN`.

**Source:** Ported from `models/gemma-4-31b/vllm/patches/vllm-gemma4-dflash-int8/`.
The `gemma4.py` model file was excluded (Qwen3 uses `qwen3_dflash.py` instead).
All other files are identical byte-for-byte.

**Status:** 🧪 Experimental — under validation.

## Three-layer fix at PR #42102

1. **`v1/core/kv_cache_utils.py`** — partition DFlash draft KV specs before
   page-size unify. Target specs (INT8 PTH) unify normally; drafter specs (BF16)
   form independent KV groups, bypassing unify entirely.

2. **`model_executor/models/qwen3_dflash.py`** — override DFlash drafter
   `cache_dtype` to `"auto"` when engine global is quantized. Drafter has
   independent KV pool post-(1) so it doesn't need target's quantized dtype.

3. **`v1/attention/backends/flash_attn.py`** — per-spec `kv_cache_dtype` reading
   when `kv_quant_mode == NONE`. Necessary because (2) puts BF16 in drafter spec
   while engine global stays INT8 PTH.

## Drop trigger

When PR #42102 + PR #41703 both land + propagate to a nightly tag:

```bash
gh api repos/vllm-project/vllm/pulls/42102 --jq '.state, .merged_at'
gh api repos/vllm-project/vllm/pulls/41703 --jq '.state, .merged_at'
```

Once both merged + propagated, bump the compose's nightly tag to a post-merge
image and drop this entire overlay directory + volume mounts from the compose.

## Rebase notes

When bumping the nightly pin:

1. Update `VERSION_PIN` to the new SHA
2. Diff each overlay file against the same path in the new nightly image:
   ```bash
   docker run --rm vllm/vllm-openai:nightly-<NEW_SHA> \
     diff /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py \
     /path/to/overlay/v1/core/kv_cache_utils.py
   ```
3. Re-apply PR #42102's three changes on top of the new base
4. Re-validate with `verify-full.sh` + `bench.sh`
