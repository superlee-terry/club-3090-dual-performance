#!/usr/bin/env bash
#
# Manage dual-3090 deployment services.
#
# Usage:
#   bash scripts/start.sh                       # start default (dflash)
#   bash scripts/start.sh start                 # same as above
#   bash scripts/start.sh start dflash          # start vLLM DFlash (FP16 KV, 185K ctx)
#   bash scripts/start.sh start dflash-int8     # start vLLM DFlash + INT8 PTH KV (262K ctx)
#   bash scripts/start.sh start moe             # start llama.cpp MoE
#   bash scripts/start.sh start moe-mtp         # start vLLM MoE AWQ + MTP-3
#   bash scripts/start.sh stop                  # stop default (dflash)
#   bash scripts/start.sh stop dflash           # stop vLLM DFlash
#   bash scripts/start.sh stop dflash-int8      # stop vLLM DFlash + INT8
#   bash scripts/start.sh stop moe              # stop llama.cpp MoE
#   bash scripts/start.sh stop moe-mtp          # stop vLLM MoE AWQ + MTP-3
#   bash scripts/start.sh restart [service]     # stop then start
#   bash scripts/start.sh status                # show all services
#   bash scripts/start.sh logs [service]        # follow logs
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Load .env if present
if [ -f "$DOCKER_ROOT/.env" ]; then
  set -a; source "$DOCKER_ROOT/.env"; set +a
fi

cd "$DOCKER_ROOT"

# --- Service definitions ---
# Each service: compose file, container name, port, model alias, env prefix
declare -A SVC_COMPOSE=(
  [dflash]="compose/dflash.yml"
  [dflash-int8]="compose/dflash-int8.yml"
  [moe]="compose/moe-llamacpp.yml"
  [moe-mtp]="compose/moe-awq-mtp.yml"
)
declare -A SVC_CONTAINER=(
  [dflash]="${CONTAINER_NAME:-vllm-qwen36-27b-dflash}"
  [dflash-int8]="${DFLASH_INT8_CONTAINER:-vllm-qwen36-27b-dflash-int8}"
  [moe]="${MOE_CONTAINER:-ik-llama-moe}"
  [moe-mtp]="${MOE_AWQ_CONTAINER:-vllm-qwen36-35b-a3b-mtp}"
)
declare -A SVC_PORT=(
  [dflash]="${PORT:-11434}"
  [dflash-int8]="${DFLASH_INT8_PORT:-11434}"
  [moe]="${MOE_PORT:-11435}"
  [moe-mtp]="${MOE_AWQ_PORT:-11436}"
)
declare -A SVC_MODEL=(
  [dflash]="${MODEL_ALIAS:-qwen3.6-27b-autoround}"
  [dflash-int8]="${MODEL_ALIAS:-qwen3.6-27b-autoround}"
  [moe]="${MOE_CONTAINER:-ik-llama-moe}"
  [moe-mtp]="${MOE_AWQ_MODEL_ALIAS:-qwen3.6-35b-a3b-awq}"
)
declare -A SVC_IMAGE=(
  [dflash]="${VLLM_IMAGE:-vllm/vllm-openai:nightly}"
  [dflash-int8]="${VLLM_IMAGE:-vllm/vllm-openai:nightly}"
  [moe]="${MOE_IMAGE:-ghcr.io/ikawrakow/ik-llama-cpp:cu13-server}"
  [moe-mtp]="${MOE_AWQ_IMAGE:-vllm/vllm-openai:nightly}"
)

DEFAULT_SVC="dflash"

resolve_service() {
  local svc="${1:-$DEFAULT_SVC}"
  case "$svc" in
    dflash|dflash-int8|moe|moe-mtp) echo "$svc" ;;
    *) echo "ERROR: Unknown service '$svc'. Use: dflash, dflash-int8, moe, moe-mtp" >&2; exit 1 ;;
  esac
}

service_url() {
  local svc="$1"
  echo "http://${BIND_HOST:-localhost}:${SVC_PORT[$svc]}"
}

# --- Actions ---

do_status() {
  echo "=== Service Status ==="
  for svc in dflash dflash-int8 moe moe-mtp; do
    local container="${SVC_CONTAINER[$svc]}"
    local url="$(service_url "$svc")"
    local model="${SVC_MODEL[$svc]}"
    echo ""
    echo "  [$svc]"
    if docker ps --format '{{.Names}}\t{{.Status}}' 2>/dev/null | grep -q "$container"; then
      docker ps --format '{{.Names}}\t{{.Status}}' | grep "$container" | sed 's/^/    /'
      if curl -sf --max-time 3 "$url/v1/models" 2>/dev/null | grep -q "$model"; then
        echo "    API:   ready ($url)"
      else
        echo "    API:   loading..."
      fi
    elif docker ps -a --format '{{.Names}}\t{{.Status}}' 2>/dev/null | grep -q "$container"; then
      docker ps -a --format '{{.Names}}\t{{.Status}}' | grep "$container" | sed 's/^/    /'
      echo "    API:   stopped"
    else
      echo "    Container not found."
    fi
  done
}

do_stop() {
  local svc="$(resolve_service "${1:-$DEFAULT_SVC}")"
  local container="${SVC_CONTAINER[$svc]}"
  local compose="${SVC_COMPOSE[$svc]}"
  echo "=== Stopping [$svc] ==="
  if ! docker ps -a --format '{{.Names}}' | grep -q "$container"; then
    echo "  Container '$container' not found."
    return 0
  fi
  docker compose --env-file .env -f "$compose" down 2>&1
  echo "  Stopped."
}

do_logs() {
  local svc="$(resolve_service "${1:-$DEFAULT_SVC}")"
  local container="${SVC_CONTAINER[$svc]}"
  if ! docker ps --format '{{.Names}}' | grep -q "$container"; then
    echo "Container '$container' is not running."
    exit 1
  fi
  docker logs -f "$container" 2>&1
}

do_start() {
  local svc="$(resolve_service "${1:-$DEFAULT_SVC}")"
  local container="${SVC_CONTAINER[$svc]}"
  local compose="${SVC_COMPOSE[$svc]}"
  local url="$(service_url "$svc")"
  local model="${SVC_MODEL[$svc]}"

  # --- Preflight ---
  echo "=== Preflight [$svc] ==="

  # 1. Check .env
  if [ ! -f ".env" ]; then
    echo "ERROR: .env not found. Run: cp .env.example .env"
    exit 1
  fi
  echo "  .env OK"

  # 2. Check model weights (service-specific)
  if [ "$svc" = "dflash" ] || [ "$svc" = "dflash-int8" ]; then
    local main_dir="${MODEL_DIR:-./models}/qwen3.6-27b/autoround-int4"
    local dflash_dir="${MODEL_DIR:-./models}/qwen3.6-27b/dflash"
    if [ ! -f "$main_dir/config.json" ] || [ "$(ls "$main_dir"/*.safetensors 2>/dev/null | wc -l)" -eq 0 ]; then
      echo "ERROR: Main model not found at $main_dir"
      echo "  Run: bash scripts/download.sh"
      exit 1
    fi
    echo "  Main model OK"
    if [ ! -f "$dflash_dir/config.json" ] || [ "$(ls "$dflash_dir"/*.safetensors 2>/dev/null | wc -l)" -eq 0 ]; then
      echo "WARNING: DFlash draft not found at $dflash_dir (TPS will be ~25 instead of ~125)"
    else
      echo "  DFlash draft OK"
    fi
    if [ "$svc" = "dflash-int8" ]; then
      local overlay_dir="./patches/vllm-qwen36-dflash-int8"
      if [ ! -d "$overlay_dir" ]; then
        echo "WARNING: DFlash+INT8 overlay not found at $overlay_dir"
        echo "  The compose will fail without the 17-file vendored overlay."
      else
        echo "  DFlash+INT8 overlay OK"
      fi
    fi
  elif [ "$svc" = "moe" ]; then
    local moe_file="${MODEL_DIR:-./models}/qwen3.6-35b-a3b/apex-quality/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf"
    if [ ! -f "$moe_file" ]; then
      echo "ERROR: MoE GGUF not found at $moe_file"
      echo "  Run: bash scripts/download.sh --moe"
      exit 1
    fi
    echo "  MoE GGUF OK ($(du -sh "$moe_file" | awk '{print $1}'))"
  elif [ "$svc" = "moe-mtp" ]; then
    local moe_awq_dir="${MODEL_DIR:-./models}/qwen3.6-35b-a3b-awq-int4"
    if [ ! -f "$moe_awq_dir/config.json" ] || [ "$(ls "$moe_awq_dir"/*.safetensors 2>/dev/null | wc -l)" -eq 0 ]; then
      echo "ERROR: MoE AWQ-INT4 not found at $moe_awq_dir"
      echo "  Run: bash scripts/download.sh --moe-awq"
      exit 1
    fi
    echo "  MoE AWQ-INT4 OK ($(du -sh "$moe_awq_dir" | awk '{print $1}'))"
  fi

  # 3. Check GPU
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
  if [ "$GPU_COUNT" -lt 2 ]; then
    echo "WARNING: Only $GPU_COUNT GPU(s) detected. TP=2 requires 2 GPUs."
  fi
  echo "  GPUs: $GPU_COUNT"

  # 4. Already running?
  if docker ps --format '{{.Names}}' | grep -q "$container"; then
    echo ""
    echo "  Container '$container' already running."
    echo "  Endpoint: $url/v1/chat/completions"
    exit 0
  fi

  echo ""

  # --- Start ---
  echo "=== Starting [$svc] ==="
  echo "  Image:     ${SVC_IMAGE[$svc]}"
  echo "  Compose:   $compose"
  echo "  Port:      ${SVC_PORT[$svc]}"
  echo "  Container: $container"
  echo ""

  docker compose --env-file .env -f "$compose" up -d 2>&1

  echo ""
  echo "Container started. Waiting for model to load..."

  # --- Wait for ready ---
  local timeout=600
  if [ "$svc" = "moe" ]; then
    timeout=300  # llama.cpp starts faster, no JIT
  fi
  local elapsed=0
  while [ $elapsed -lt $timeout ]; do
    if ! docker ps --format '{{.Names}}' | grep -q "$container"; then
      echo ""
      echo "ERROR: Container exited unexpectedly. Check logs:"
      echo "  docker logs $container"
      exit 1
    fi

    if curl -sf --max-time 3 "$url/v1/models" 2>/dev/null | grep -q .; then
      echo ""
      echo "=== Service [$svc] ready ==="
      echo "  Endpoint: $url/v1/chat/completions"
      echo ""
      echo "Quick test:"
      echo "  curl -sf $url/v1/chat/completions \\"
      echo "    -H 'Content-Type: application/json' \\"
      echo "    -d '{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":100}'"
      exit 0
    fi

    elapsed=$((elapsed + 5))
    local docker_log=$(docker logs --tail 1 "$container" 2>&1)
    if echo "$docker_log" | grep -qi "load\|warm\|init"; then
      echo -n "."
    elif echo "$docker_log" | grep -qi "error\|failed\|fatal"; then
      echo ""
      echo "ERROR during startup. Check logs:"
      echo "  docker logs $container"
      exit 1
    fi
  done

  echo ""
  echo "TIMEOUT: Service not ready after ${timeout}s. Check logs:"
  echo "  docker logs $container"
  exit 1
}

# --- Dispatch ---
ACTION="${1:-start}"
shift 2>/dev/null || true

case "$ACTION" in
  start)   do_start "$@" ;;
  stop)    do_stop "$@" ;;
  restart) svc="${1:-$DEFAULT_SVC}"; do_stop "$svc"; echo ""; do_start "$svc" ;;
  status)  do_status ;;
  logs)    do_logs "$@" ;;
  *)
    echo "Usage: bash scripts/start.sh {start|stop|restart|status|logs} [service]"
    echo ""
    echo "Commands:"
    echo "  start [dflash|dflash-int8|moe|moe-mtp]  Start service with preflight check"
    echo "  stop [dflash|dflash-int8|moe|moe-mtp]   Stop service"
    echo "  restart [service]                        Stop then start"
    echo "  status                                   Show all services"
    echo "  logs [service]                           Follow container logs"
    echo ""
    echo "Services:"
    echo "  dflash       Qwen3.6-27B vLLM DFlash (FP16 KV, 185K ctx) [default]"
    echo "  dflash-int8  Qwen3.6-27B vLLM DFlash + INT8 PTH KV (262K ctx)"
    echo "  moe          Qwen3.6-35B-A3B llama.cpp MoE"
    echo "  moe-mtp      Qwen3.6-35B-A3B vLLM AWQ + MTP-3"
    exit 1
    ;;
esac
