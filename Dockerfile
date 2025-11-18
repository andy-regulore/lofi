# Multi-stage Docker build for lo-fi music generator

# Stage 1: Base image with system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    fluidsynth \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: Development image
FROM base as development

# Install development dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY . .

# Install package in editable mode
RUN pip install -e .

CMD ["/bin/bash"]

# Stage 3: Production image
FROM base as production

# Copy only requirements first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config.yaml pyproject.toml README.md ./

# Install package
RUN pip install --no-cache-dir -e .

# Create directories for data and models
RUN mkdir -p /app/data/midi /app/data/tokens /app/data/datasets \
    /app/models /app/output

# Set up non-root user for security
RUN useradd -m -u 1000 lofi && \
    chown -R lofi:lofi /app

USER lofi

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "scripts.04_generate"]

# Stage 4: Training image with GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as training

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    ffmpeg \
    fluidsynth \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy application
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config.yaml pyproject.toml ./

# Install package
RUN pip3 install --no-cache-dir -e .

# Create directories
RUN mkdir -p /app/data /app/models /app/output

# Set up user
RUN useradd -m -u 1000 lofi && \
    chown -R lofi:lofi /app

USER lofi

CMD ["python3", "-m", "scripts.03_train"]
