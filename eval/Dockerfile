FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd -g 10001 appuser && useradd -u 10001 -g appuser -d /app -s /sbin/nologin appuser

COPY indexer/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY indexer/pipeline ./indexer/pipeline
COPY eval ./eval

RUN mkdir -p /app/eval/results && chown -R appuser:appuser /app

USER appuser

CMD ["python", "eval/run_all.py"]
