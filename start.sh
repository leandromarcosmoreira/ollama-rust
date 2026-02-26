#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OLLAMA_MODELS_LIST=$(grep -E "^OLLAMA_MODELS=" ../.env 2>/dev/null | cut -d'=' -f2 || echo "")

if [ -z "$OLLAMA_MODELS_LIST" ]; then
    echo "Warning: OLLAMA_MODELS not found in ../.env"
    OLLAMA_MODELS_LIST="tinyllama"
fi

echo "=== Starting Ollama Rust ==="
echo "Models to sync: $OLLAMA_MODELS_LIST"

export OLLAMA_MODELS_LIST

docker compose up -d "$@"

echo ""
sleep 3
docker compose logs --tail 20
