FROM python:3.11-slim AS wheel-builder

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.11-slim AS runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/models/model.joblib

COPY requirements.txt .
COPY --from=wheel-builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY model_core.py train_model.py server.py ./
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
