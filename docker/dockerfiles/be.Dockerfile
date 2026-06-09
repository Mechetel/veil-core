# syntax=docker/dockerfile:1
# Development base image for veil-core. Application code is bind-mounted and Python
# dependencies are installed at runtime into a persisted /opt/venv volume (see
# docker/dev.yml) — mirroring veil-web's be.Dockerfile + be_gems pattern.

ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libraries needed by torch / pillow / scikit-image at runtime + build tools.
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y \
      build-essential git curl libjpeg-dev zlib1g-dev libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists /var/cache/apt/archives

WORKDIR /app
