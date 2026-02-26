#!/bin/bash
set -e

echo "=== Ollama Rust Stack Starting ==="

mkdir -p /home/ollama/.ollama/models

echo "[Entry] Checking existing models on disk..."
if [ -d "/home/ollama/.ollama/models" ]; then
    find /home/ollama/.ollama/models -name "manifest.json" -type f 2>/dev/null | while read manifest; do
        model_dir=$(dirname "$manifest")
        model_name=$(basename "$model_dir" | sed 's/--/\//g')
        echo "[Entry]   Found: $model_name"
    done
fi

echo "[Entry] Starting Ollama server..."
/usr/local/bin/ollama serve &
SERVER_PID=$!

MAX_WAIT=30
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s -f http://localhost:11434/api/health > /dev/null 2>&1; then
        echo "[Entry] Ollama server is healthy"
        break
    fi
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ $WAIT_COUNT -eq $MAX_WAIT ]; then
    echo "[Entry] Warning: Ollama server health check timed out"
fi

if [ -n "$OLLAMA_MODELS_LIST" ]; then
    echo "[Entry] Starting Healthchecker..."
    echo "[Entry] Models to sync: $OLLAMA_MODELS_LIST"
    export OLLAMA_HOST=http://localhost:11434
    /usr/local/bin/ollama-healthchecker &
    HEALTHCHECKER_PID=$!
else
    echo "[Entry] OLLAMA_MODELS_LIST not set, healthchecker disabled"
    echo "[Entry] Set OLLAMA_MODELS_LIST to enable auto-sync (e.g., OLLAMA_MODELS_LIST=tinyllama,llama3.2)"
fi

cleanup() {
    echo "[Entry] Shutting down..."
    kill $SERVER_PID 2>/dev/null || true
    kill $HEALTHCHECKER_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

echo "[Entry] Stack started successfully"
wait $SERVER_PID
