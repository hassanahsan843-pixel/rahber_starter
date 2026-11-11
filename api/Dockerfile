# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Refresh Debian security updates in the slim base image (including zlib).
RUN apt-get update && apt-get upgrade -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Create a dedicated non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Install dependencies first for efficient layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip "setuptools==78.1.1" && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app ./app

# Set ownership and permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user (section 4.7)
USER appuser

EXPOSE 8000

# Exec form CMD (section 4.7)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
