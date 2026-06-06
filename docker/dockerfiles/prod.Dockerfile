# syntax=docker/dockerfile:1
# check=error=true

# Production image for the Veil Core FastAPI service. Use with Kamal or by hand:
#   docker build -f docker/dockerfiles/prod.Dockerfile -t veil-core .
#   docker run -d -p 8000:8000 -e VEIL_CORE_TOKEN=<token> --name veil-core veil-core

ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim AS base

# App lives here
WORKDIR /app

# Don't write .pyc files; flush stdout/stderr immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install curl for container healthchecks
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y curl && \
    rm -rf /var/lib/apt/lists /var/cache/apt/archives

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app

# Run as a non-root user for security
RUN useradd --create-home --uid 1000 veil && \
    chown -R veil:veil /app
USER 1000:1000

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
