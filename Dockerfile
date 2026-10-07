FROM python:3.11-slim

WORKDIR /app

# The package and its dependencies; config.yaml stays outside it, and the
# paths in it (data_cache/, outputs/) are relative to it, so they land in /app.
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install --no-cache-dir .

COPY config.yaml .

# data_cache/ holds downloaded prices and fundamentals; mount it as a volume so
# reruns reuse it instead of re-downloading. outputs/ holds the results.
VOLUME ["/app/data_cache", "/app/outputs"]

ENTRYPOINT ["factor-backtest"]
CMD ["run", "--config", "config.yaml"]
