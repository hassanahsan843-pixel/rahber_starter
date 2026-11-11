# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for PDF processing & OCR (poppler-utils, tesseract-ocr)
RUN apt-get update && \
    apt-get install -y --no-install-recommends --only-upgrade zlib1g && \
    apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a dedicated non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade \
        pip \
        "setuptools==84.2.1" \
        "wheel==0.48.0" \
        "jaraco.context==6.1.2" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check

# Copy the indexer pipeline. Kubernetes supplies the corpus through a read-only
# ConfigMap in CI/dev or a PVC in prod.
COPY pipeline ./pipeline

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user (section 4.7)
USER appuser

# Exec form ENTRYPOINT (section 4.7)
ENTRYPOINT ["python", "-m", "pipeline.run"]
