#!/bin/bash
set -e

# Start ollama in background
ollama serve &
OLLAMA_PID=$!

# Wait for ollama to be ready
echo "Waiting for Ollama to start..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready"
        break
    fi
    sleep 1
done

# Preload model into memory
echo "Preloading exaone3.5:7.8b model (this may take 60+ seconds)..."
curl -s http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"exaone3.5:7.8b","messages":[{"role":"user","content":"안녕"}],"stream":false}' \
    > /dev/null
echo "Model preloaded successfully - ready for requests"

# Wait for ollama process
wait $OLLAMA_PID
