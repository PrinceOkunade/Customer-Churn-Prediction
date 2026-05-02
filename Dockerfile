FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY src/ /app/src/
COPY models/ /app/models/

WORKDIR /app/src

ENV PORT=8080
EXPOSE 8080

# shell form so $PORT (set by Cloud Run at runtime) expands; exec keeps uvicorn as PID 1
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT}
