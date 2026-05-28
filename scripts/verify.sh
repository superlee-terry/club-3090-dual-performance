#!/usr/bin/env bash
#
# Quick health check for the Qwen3.6-27B DFlash deployment.
#
# Usage: bash scripts/verify.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Load .env if present
if [ -f "$DOCKER_ROOT/.env" ]; then
  set -a; source "$DOCKER_ROOT/.env"; set +a
fi

PORT="${PORT:-8012}"
HOST="${BIND_HOST:-localhost}"
URL="http://${HOST}:${PORT}"
PASS=0
FAIL=0

echo "=== Verifying deployment at $URL ==="
echo ""

# 1. Container running
echo -n "[1/5] Container running ... "
if docker ps --format '{{.Names}}' | grep -q "${CONTAINER_NAME:-vllm-qwen36-27b-dflash}"; then
  echo "OK"; ((PASS++))
else
  echo "FAIL (container not found)"; ((FAIL++))
fi

# 2. /v1/models endpoint
echo -n "[2/5] /v1/models ... "
MODELS=$(curl -sf --max-time 10 "$URL/v1/models" 2>/dev/null || echo "")
if echo "$MODELS" | grep -q "${MODEL_ALIAS:-qwen3.6-27b-autoround}"; then
  echo "OK"; ((PASS++))
else
  echo "FAIL (endpoint unreachable or model not loaded)"; ((FAIL++))
fi

# 3. Chat completion (short prompt)
echo -n "[3/5] Chat completion ... "
RESPONSE=$(curl -sf --max-time 30 "$URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"${MODEL_ALIAS:-qwen3.6-27b-autoround}","messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"max_tokens":20}' 2>/dev/null || echo "")
if echo "$RESPONSE" | grep -q '"choices"'; then
  CONTENT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
  echo "OK ($CONTENT)"; ((PASS++))
else
  echo "FAIL (no response)"; ((FAIL++))
fi

# 4. Tool call support
echo -n "[4/5] Tool call support ... "
TOOL_RESP=$(curl -sf --max-time 30 "$URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"${MODEL_ALIAS:-qwen3.6-27b-autoround}","messages":[{"role":"user","content":"What is the weather in Tokyo?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"tool_choice":"auto","max_tokens":100}' 2>/dev/null || echo "")
if echo "$TOOL_RESP" | grep -q "tool_calls\|get_weather"; then
  echo "OK"; ((PASS++))
else
  echo "FAIL (tool call not working)"; ((FAIL++))
fi

# 5. DFlash spec-decode active (check logs for acceptance rate)
echo -n "[5/5] DFlash spec-decode ... "
CONTAINER="${CONTAINER_NAME:-vllm-qwen36-27b-dflash}"
LOGS=$(docker logs "$CONTAINER" --tail 100 2>&1 || echo "")
if echo "$LOGS" | grep -qi "speculat\|dflash\|draft"; then
  echo "OK (draft model loaded)"; ((PASS++))
else
  echo "WARN (no draft evidence in recent logs)"; ((FAIL++))
fi

# --- Result ---
echo ""
if [ "$FAIL" -eq 0 ]; then
  echo "=== ALL $PASS CHECKS PASSED ==="
else
  echo "=== $PASS passed, $FAIL failed ==="
fi

# Print endpoint info
echo ""
echo "Endpoint: $URL/v1/chat/completions"
echo "Model:    ${MODEL_ALIAS:-qwen3.6-27b-autoround}"
echo ""
echo "Quick test:"
echo "  curl -sf $URL/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${MODEL_ALIAS:-qwen3.6-27b-autoround}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":100}'"
