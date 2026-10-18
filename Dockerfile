# ==============================================================================
# SensorTrust Production ASGI Backend Dockerfile
# Optimized for Google Cloud Run, Render, Railway, AWS ECS, or Fly.io
# ==============================================================================

FROM python:3.11-slim

# Set environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    ENVIRONMENT=production

# Install system dependencies (libgomp1 is strictly required by LightGBM on Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-trained model artifact
COPY core.py .
COPY api_server.py .
COPY ai_agent.py .
COPY voice_system.py .
COPY evaluate.py .
COPY model.joblib .

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Container Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Production ASGI Entrypoint with proxy headers and multi-platform port binding
CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips '*'"]
