FROM python:3.12-slim

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download turn detector model files
RUN python agent/agent.py download-files

ENV PYTHONUNBUFFERED=1

# Default to agent.py with start command
CMD ["python", "agent/agent.py", "start"]
