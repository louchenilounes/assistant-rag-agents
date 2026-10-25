FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY src/ src/
COPY scripts/build_index.py scripts/
COPY app/ app/
COPY data/raw/ data/raw/

RUN useradd --create-home appuser && mkdir storage && chown appuser storage
USER appuser

EXPOSE 8000 8501
