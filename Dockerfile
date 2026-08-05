FROM python:3.11-slim AS builder


COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache

FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/nampq11/rag"

WORKDIR /app

COPY --from=builder /app/.venv ./.venv

COPY app ./app
COPY main.py ./

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["python", "main.py"]
