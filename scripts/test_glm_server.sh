#!/bin/bash
# Start llama-server with GLM-4.7-Flash in background, then run curl test in foreground

MODEL_REPO="bartowski/zai-org_GLM-4.7-Flash-GGUF"
MODEL_FILE="zai-org_GLM-4.7-Flash-Q4_K_M.gguf"

# Bench findings so far (prompt processing pp1024 on BLAS/Accelerate, 15 threads):
# best seen: n_batch=128, n_ubatch=128
N_BATCH=128
N_UBATCH=128

# Kill any existing llama-server
pkill -f "llama-server" 2>/dev/null
sleep 1

# Start server in background
echo "Starting llama-server..."
./build/bin/llama-server \
  --hf-repo "$MODEL_REPO" \
  --hf-file "$MODEL_FILE" \
  -ngl 0 \
  -b $N_BATCH \
  -ub $N_UBATCH \
  -c 8192 \
  --threads 15 \
  --host 127.0.0.1 \
  --port 8080 \
  --no-warmup \
  &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server and model to be fully ready
echo "Waiting for server and model to load..."
for i in {1..120}; do
  HEALTH=$(curl -s http://127.0.0.1:8080/health 2>/dev/null)
  if echo "$HEALTH" | grep -q '"status":"ok"'; then
    echo "Server and model ready!"
    break
  fi
  echo "  ($i) Still loading..."
  sleep 2
done

# Run curl test in foreground
echo "Running test with max_tokens=500..."
time curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a haiku about AI"}],"max_tokens":500,"temperature":0.1}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); r=d['choices'][0]['message']; print(f\"Tokens: {d['usage']['completion_tokens']}\n\nReasoning:\n{r.get('reasoning_content','none')[:500]}...\n\nAnswer:\n{r.get('content','none')}\")"

echo ""
echo "Test complete. Server still running (PID: $SERVER_PID)"
echo "Stop server with: kill $SERVER_PID"
