#!/usr/bin/env bash
#
# Download model weights for Qwen3.6-27B DFlash dual-3090 deployment.
#
# Strategy: ModelScope first, HuggingFace fallback.
#   ModelScope (modelscope.cn) is faster in mainland China.
#   If a model isn't found on ModelScope, falls back to HuggingFace automatically.
#
# Usage:
#   bash scripts/download.sh              # download main + DFlash (ModelScope → HF)
#   bash scripts/download.sh --main-only  # skip DFlash draft and MoE
#   bash scripts/download.sh --moe        # download MoE GGUF only
#   bash scripts/download.sh --moe-awq    # download MoE AWQ-INT4 only
#   bash scripts/download.sh --all        # download everything (main + DFlash + MoE + MoE AWQ)
#   bash scripts/download.sh --source hf  # force HuggingFace only
#   bash scripts/download.sh --source ms  # force ModelScope only
#   MODEL_DIR=/data/models bash scripts/download.sh
#
# Prerequisites:
#   modelscope:  pip install modelscope
#   huggingface: pip install huggingface_hub
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Load .env if present
if [ -f "$DOCKER_ROOT/.env" ]; then
  set -a; source "$DOCKER_ROOT/.env"; set +a
fi

MODEL_DIR="${MODEL_DIR:-$DOCKER_ROOT/models}"

# --- Repo mapping ---
# Main model: Intel's official upload on ModelScope, Lorbus mirror on HF.
# DFlash draft: same repo ID on both platforms.
MS_MAIN_REPO="hf/Lorbus-Qwen3.6-27B-int4-AutoRound"
HF_MAIN_REPO="Lorbus/Qwen3.6-27B-int4-AutoRound"
MS_DFLASH_REPO="z-lab/Qwen3.6-27B-DFlash"
HF_DFLASH_REPO="z-lab/Qwen3.6-27B-DFlash"

# MoE GGUF: APEX Quality quantization for Qwen3.6-35B-A3B
MS_MOE_REPO="hf/mudler-Qwen3.6-35B-A3B-APEX-MTP-GGUF"
HF_MOE_REPO="mudler/Qwen3.6-35B-A3B-APEX-MTP-GGUF"
MOE_FILE="Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf"

# MoE AWQ: vLLM AWQ-INT4 for Qwen3.6-35B-A3B
HF_MOE_AWQ_REPO="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"

MAIN_DIR="$MODEL_DIR/qwen3.6-27b/autoround-int4"
DFLASH_DIR="$MODEL_DIR/qwen3.6-27b/dflash"
MOE_DIR="$MODEL_DIR/qwen3.6-35b-a3b/apex-quality"
MOE_AWQ_DIR="$MODEL_DIR/qwen3.6-35b-a3b-awq-int4"

SKIP_DFLASH=""
DOWNLOAD_MOE=""
DOWNLOAD_MOE_AWQ=""
SOURCE="auto"  # auto | ms | hf

# --- Parse args ---
while [ $# -gt 0 ]; do
  case "$1" in
    --main-only) SKIP_DFLASH=1 ;;
    --moe)       DOWNLOAD_MOE=1 ;;
    --moe-awq)   DOWNLOAD_MOE_AWQ=1 ;;
    --all)       DOWNLOAD_MOE=1; DOWNLOAD_MOE_AWQ=1 ;;
    --source)    shift; SOURCE="${1:-auto}" ;;
  esac
  shift
done

# --- Download functions ---

download_modelscope() {
  local repo_id="$1"
  local target_dir="$2"
  echo "  Source: ModelScope ($repo_id)"
  python3 -c "
from modelscope import snapshot_download
snapshot_download('$repo_id', local_dir='$target_dir')
"
}

download_huggingface() {
  local repo_id="$1"
  local target_dir="$2"
  echo "  Source: HuggingFace ($repo_id)"
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo_id', local_dir='$target_dir' ${HF_TOKEN:+, token='$HF_TOKEN'})
"
}

download_model() {
  local label="$1"
  local ms_repo="$2"
  local hf_repo="$3"
  local target_dir="$4"

  if [ -f "$target_dir/config.json" ] && [ "$(ls "$target_dir"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  SKIP: $label already exists at $target_dir"
    return 0
  fi

  mkdir -p "$target_dir"

  if [ "$SOURCE" = "hf" ]; then
    download_huggingface "$hf_repo" "$target_dir"
  elif [ "$SOURCE" = "ms" ]; then
    download_modelscope "$ms_repo" "$target_dir"
  else
    # auto: try ModelScope first, fall back to HF
    echo "  Trying ModelScope..."
    if ! download_modelscope "$ms_repo" "$target_dir" 2>/dev/null; then
      echo "  ModelScope unavailable or repo not found, falling back to HuggingFace..."
      download_huggingface "$hf_repo" "$target_dir"
    fi
  fi
  echo "  DONE: $label downloaded."
}

# --- Preflight ---
echo "=== Preflight ==="

DISK_AVAIL=$(df -BG "$MODEL_DIR" 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G')
if [ "${DISK_AVAIL:-0}" -lt 20 ]; then
  echo "WARNING: Only ${DISK_AVAIL:-?} GB available at $MODEL_DIR. Need ~20 GB."
fi

# Check SDK availability
has_ms=false
has_hf=false
python3 -c "from modelscope import snapshot_download" 2>/dev/null && has_ms=true
command -v hf &>/dev/null && has_hf=true
python3 -c "from huggingface_hub import snapshot_download" 2>/dev/null && has_hf=true

if [ "$SOURCE" = "ms" ] && [ "$has_ms" = "false" ]; then
  echo "ERROR: modelscope not found (forced --source ms). Install with: pip install modelscope"
  exit 1
fi
if [ "$SOURCE" = "hf" ] && [ "$has_hf" = "false" ]; then
  echo "ERROR: huggingface_hub not found (forced --source hf). Install with: pip install huggingface_hub"
  exit 1
fi
if [ "$has_ms" = "false" ] && [ "$has_hf" = "false" ]; then
  echo "ERROR: No download SDK found. Install at least one:"
  echo "  pip install modelscope       # ModelScope (faster in China)"
  echo "  pip install huggingface_hub  # HuggingFace"
  exit 1
fi

if [ "$has_ms" = "true" ]; then echo "  SDK: modelscope ✓"; fi
if [ "$has_hf" = "true" ]; then echo "  SDK: huggingface_hub ✓"; fi
echo "  Source: $SOURCE"

mkdir -p "$MAIN_DIR" "$DFLASH_DIR" "$MOE_DIR" "$MOE_AWQ_DIR"

# --- Download main model ---
echo ""
echo "=== Downloading main model ==="
download_model "Main model" "$MS_MAIN_REPO" "$HF_MAIN_REPO" "$MAIN_DIR"

# --- Download DFlash draft ---
if [ -n "$SKIP_DFLASH" ]; then
  echo ""
  echo "=== Skipping DFlash draft (--main-only) ==="
else
  echo ""
  echo "=== Downloading DFlash draft model ==="
  download_model "DFlash draft" "$MS_DFLASH_REPO" "$HF_DFLASH_REPO" "$DFLASH_DIR"
fi

# --- Download MoE GGUF ---
if [ -n "$DOWNLOAD_MOE" ]; then
  echo ""
  echo "=== Downloading MoE GGUF ==="
  if [ -f "$MOE_DIR/$MOE_FILE" ]; then
    echo "  SKIP: MoE GGUF already exists at $MOE_DIR/$MOE_FILE"
  else
    if [ "$SOURCE" = "hf" ]; then
      echo "  Source: HuggingFace ($HF_MOE_REPO)"
      if command -v hf &>/dev/null; then
        hf download "$HF_MOE_REPO" "$MOE_FILE" --local-dir "$MOE_DIR" ${HF_TOKEN:+--token "$HF_TOKEN"}
      else
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$HF_MOE_REPO', '$MOE_FILE', local_dir='$MOE_DIR' ${HF_TOKEN:+, token='$HF_TOKEN'})
"
      fi
    elif [ "$SOURCE" = "ms" ]; then
      echo "  Source: ModelScope ($MS_MOE_REPO)"
      python3 -c "
from modelscope import snapshot_download
snapshot_download('$MS_MOE_REPO', local_dir='$MOE_DIR', allow_patterns=['$MOE_FILE'])
"
    else
      echo "  Trying ModelScope..."
      if ! python3 -c "
from modelscope import snapshot_download
snapshot_download('$MS_MOE_REPO', local_dir='$MOE_DIR', allow_patterns=['$MOE_FILE'])
" 2>/dev/null; then
        echo "  ModelScope unavailable, falling back to HuggingFace..."
        if command -v hf &>/dev/null; then
          hf download "$HF_MOE_REPO" "$MOE_FILE" --local-dir "$MOE_DIR" ${HF_TOKEN:+--token "$HF_TOKEN"}
        else
          python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$HF_MOE_REPO', '$MOE_FILE', local_dir='$MOE_DIR' ${HF_TOKEN:+, token='$HF_TOKEN'})
"
        fi
      fi
    fi
    echo "  DONE: MoE GGUF downloaded."
  fi
fi

# --- Download MoE AWQ-INT4 ---
if [ -n "$DOWNLOAD_MOE_AWQ" ]; then
  echo ""
  echo "=== Downloading MoE AWQ-INT4 ==="
  download_model "MoE AWQ-INT4" "" "$HF_MOE_AWQ_REPO" "$MOE_AWQ_DIR"
fi

# --- Summary ---
echo ""
echo "=== Download summary ==="
MAIN_SIZE=$(du -sh "$MAIN_DIR" 2>/dev/null | awk '{print $1}' || echo "?")
DFLASH_SIZE=$(du -sh "$DFLASH_DIR" 2>/dev/null | awk '{print $1}' || echo "?")
echo "  Main model:    $MAIN_DIR ($MAIN_SIZE)"
echo "  DFlash draft:  $DFLASH_DIR ($DFLASH_SIZE)"
if [ -f "$MOE_DIR/$MOE_FILE" ]; then
  MOE_SIZE=$(du -sh "$MOE_DIR" 2>/dev/null | awk '{print $1}' || echo "?")
  echo "  MoE GGUF:      $MOE_DIR ($MOE_SIZE)"
fi
if [ -f "$MOE_AWQ_DIR/config.json" ]; then
  AWQ_SIZE=$(du -sh "$MOE_AWQ_DIR" 2>/dev/null | awk '{print $1}' || echo "?")
  echo "  MoE AWQ-INT4:  $MOE_AWQ_DIR ($AWQ_SIZE)"
fi
echo ""
echo "Next step:"
echo "  cd $DOCKER_ROOT"
echo "  bash scripts/start.sh start dflash       # vLLM DFlash (FP16 KV, 185K ctx)"
echo "  bash scripts/start.sh start dflash-int8  # vLLM DFlash + INT8 PTH KV (262K ctx)"
echo "  bash scripts/start.sh start moe          # llama.cpp MoE"
echo "  bash scripts/start.sh start moe-mtp      # vLLM MoE AWQ + MTP-3"
