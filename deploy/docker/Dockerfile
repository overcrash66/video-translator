# Use an official PyTorch image with CUDA support
# This image includes CUDA 12.1 and cuDNN 9
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch dependencies explicitly first (to ensure correct CUDA version)
# The base image already has torch, but we want to ensure compatibility with audio/vision
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PaddlePaddle GPU (requires specific CUDA version matching)
RUN pip install --no-cache-dir paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Copy requirements
COPY deploy/docker/requirements.docker.txt /app/requirements.txt

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/temp /app/output /app/models

# Copy source code
COPY . /app

# Expose Gradio port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
