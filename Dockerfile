FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY main.py ./

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["python", "main.py"]
