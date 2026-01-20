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

# Copy startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Expose Ollama API port
EXPOSE 11434

# Override ENTRYPOINT and use bash to run startup script
ENTRYPOINT ["/bin/bash"]
CMD ["/start.sh"]
