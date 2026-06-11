# syntax=docker/dockerfile:1
# check=error=true

# Production image for veil-core (FastAPI API + Celery workers share this image;
# the worker roles override the command in config/deploy.yml). Weights are NOT
# baked in — they are mounted at /app/weights from a persistent volume.
#
# CPU build (default):  docker build -f docker/dockerfiles/prod.Dockerfile -t veil-core .
# GPU build:            ... --build-arg TORCH_INDEX=cu128  (any index from
#                        https://pytorch.org/get-started/locally/ works)

# 3.12 is required: the pinned NumPy 1.26.4 / scikit-image 0.22.0 stack ships no
# Python 3.13 wheels (pip would fall back to source builds and fail).
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app


# Runtime system libraries for torch / pillow / scikit-image + curl for healthcheck.
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y \
      curl libjpeg62-turbo zlib1g libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists /var/cache/apt/archives

# Python deps. torch wheel: latest from the cpu index by default; pass
# --build-arg TORCH_INDEX=cu128 (etc.) for a GPU build. torch/torchvision are
# installed together so pip resolves a matching pair.
ARG TORCH_INDEX=cpu
COPY requirements.txt ./
RUN pip install -r requirements.txt && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_INDEX}

# Application code + vendored ML packages (needed to unpickle .steg checkpoints).
COPY app ./app
COPY steganogan ./steganogan
COPY steganalyzers ./steganalyzers

# Non-root runtime user.
RUN useradd --create-home --uid 1000 veil && chown -R veil:veil /app
USER 1000:1000

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
