# syntax=docker/dockerfile:1
# Development base image for veil-core. Application code is bind-mounted and Python
# dependencies are installed at runtime into a persisted /opt/venv volume (see
# docker/dev.yml) — mirroring veil-web's be.Dockerfile + be_gems pattern.

# 3.12 is required: the pinned NumPy 1.26.4 / scikit-image 0.22.0 stack ships no
# Python 3.13 wheels (pip would fall back to source builds and fail).
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libraries needed by torch / pillow / scikit-image at runtime + build tools.
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y \
      build-essential git curl libjpeg-dev zlib1g-dev libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists /var/cache/apt/archives

WORKDIR /app
