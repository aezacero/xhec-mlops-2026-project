FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /bin/uv

WORKDIR /app

# Install the dependencies first so this layer is cached when only the code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:$PATH"

COPY src/ src/
COPY bin/ bin/

# The modelling modules import each other by name (e.g. `from preprocessing import ...`)
ENV PYTHONPATH=/app/src/modelling
# The browser reaches the Prefect API through the host port 4200 (see `docker run`)
ENV PREFECT_UI_API_URL=http://localhost:4200/api
# Avoids "database is locked" errors from Prefect telemetry at startup
ENV PREFECT_SERVER_ANALYTICS_ENABLED=false

EXPOSE 8001 4201
CMD ["bash", "bin/run_services.sh"]
