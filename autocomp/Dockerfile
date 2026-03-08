FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace/autocomp

# Install dependencies first (cache layer)
COPY autocomp/pyproject.toml .
RUN uv sync

# Copy autocomp source
COPY autocomp/ .

# Competition workspace and kaggle credentials are mounted at runtime.
# ANTHROPIC_API_KEY is passed as an env var.

ENTRYPOINT ["uv", "run", "python", "run_session.py"]
