# Use an official NVIDIA CUDA image with Developer Tools (nvcc) for compiling deps like FlashAttention/Deepspeed if needed
# CUDA 12.4 is compatible with Torch 2.4+ and RTX 50 series
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Install system dependencies
# Added curl, python3-venv for venv support if needed (though typically not used inside docker, good for hygiene)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    curl \
    python3.10 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Copy the requirements file from the build context (project root)
# The build command is run from root, so path is deploy/docker/requirements.docker.txt
COPY deploy/docker/requirements.docker.txt /app/requirements.txt

# Install Python dependencies
# Note: Torch and Paddle are handled inside requirements.docker.txt now with specific indices
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/temp /app/output /app/models

# Copy source code
COPY . /app

# Download models during build to ensure they are baked into the image
RUN python3 scripts/download_models.py

# Expose Gradio port
EXPOSE 7860

# Command to run the application
CMD ["python3", "app.py"]
