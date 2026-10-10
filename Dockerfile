# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first, so code changes don't invalidate the layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 sigil && mkdir -p /data && chown sigil /data
USER sigil
ENV SIGIL_DATA_DIR=/data PATH="/app/.venv/bin:$PATH"
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["sigil", "serve", "--host", "0.0.0.0", "--port", "8000"]
