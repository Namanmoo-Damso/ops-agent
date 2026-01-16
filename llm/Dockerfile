# Ollama with EXAONE 3.5 Korean model (LG AI Research)
# Tool calling supported, optimized for Korean

FROM ollama/ollama:latest

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Environment variables for model configuration
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_KEEP_ALIVE=24h

# Download EXAONE 3.5 at build time (official Ollama model)
RUN ollama serve & \
    pid=$! && \
    sleep 5 && \
    ollama pull exaone3.5:7.8b && \
    kill $pid

# Create startup script that preloads model
RUN cat > /start.sh <<'SCRIPT'
#!/bin/bash
set -e

# Start ollama in background
ollama serve &
OLLAMA_PID=$!

# Wait for ollama to be ready
echo "Waiting for Ollama to start..."
for i in {1..30}; do
  if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
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
SCRIPT

RUN chmod +x /start.sh

# Expose Ollama API port
EXPOSE 11434

# Override ENTRYPOINT and use bash to run startup script
ENTRYPOINT ["/bin/bash"]
CMD ["/start.sh"]
