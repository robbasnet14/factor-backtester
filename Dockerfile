FROM python:3.11-slim

WORKDIR /app

# scripts/run_backtest.py imports from src/, which lives at the project root.
ENV PYTHONPATH=/app

# Install dependencies in their own layer so code edits don't trigger a reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/
COPY scripts/ scripts/

# data_cache/ holds downloaded prices and fundamentals; mount it as a volume so
# reruns reuse it instead of re-downloading. outputs/ holds the results.
VOLUME ["/app/data_cache", "/app/outputs"]

ENTRYPOINT ["python", "scripts/run_backtest.py"]
CMD ["--config", "config.yaml"]
