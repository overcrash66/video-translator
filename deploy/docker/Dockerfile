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

# Install Python dependencies in smaller groups to avoid pip's "resolution-too-deep" error
# This is necessary due to the complex dependency graph of ML/audio packages

# 1. Install PyTorch separately (must be first for CUDA support)
RUN pip install --no-cache-dir "torch>=2.7.0" "torchvision>=0.20.0" "torchaudio>=2.7.0" --extra-index-url https://download.pytorch.org/whl/cu128

# 2. Install core dependencies (numpy, scipy, etc. - these are foundational)
RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    "scipy<1.13.0" \
    "librosa>=0.10.0" \
    "numba>=0.57.0" \
    "llvmlite>=0.40.0" \
    "pandas<2.0.0"

# 3. Install ML/Transformers stack
RUN pip install --no-cache-dir \
    "transformers>=4.48.0" \
    "tokenizers>=0.21.0" \
    "accelerate>=0.33.0" \
    "huggingface_hub>=0.25.1" \
    "safetensors>=0.5.2" \
    "diffusers==0.29.0" \
    "torchmetrics>=1.6.1" \
    "einops>=0.8.0"

# 4. Install audio processing stack
RUN pip install --no-cache-dir \
    "pyannote.audio==4.0.1" \
    "speechbrain==1.0.3" \
    "diart==0.9.0" \
    "demucs==4.0.1" \
    "pyworld==0.3.4" \
    "pydub==0.25.1" \
    "soundfile>=0.13.1" \
    "faster-whisper==1.1.1"

# 5. Install TTS engines
RUN pip install --no-cache-dir \
    "edge-tts==6.1.10" \
    "piper-tts==1.2.0" \
    "f5-tts" \
    "TTS==0.22.0"

# 6. Install video/face processing
RUN pip install --no-cache-dir \
    "opencv-python-headless==4.10.0.84" \
    "face-alignment==1.4.1" \
    "basicsr==1.4.2" \
    "kornia==0.7.3" \
    "gfpgan==1.3.8" \
    "insightface>=0.7.3"

# 7. Install OCR and remaining packages
RUN pip install --no-cache-dir \
    "paddlepaddle-gpu==2.6.1" \
    "paddleocr==2.8.1" \
    "easyocr==1.7.2" \
    "langdetect==1.0.9"

# 8. Install web/API stack and utilities
RUN pip install --no-cache-dir \
    "gradio>=5.7.1" \
    "pydantic==2.10.6" \
    "fastapi>=0.112.0" \
    "uvicorn>=0.30.0" \
    "python-multipart" \
    "deep-translator" \
    "python-dotenv==1.0.1" \
    "pyarrow==20.0.0" \
    "ffmpeg-python==0.2.0" \
    "tqdm==4.66.4" \
    "bitsandbytes==0.43.1" \
    "sentencepiece==0.2.0" \
    "protobuf==5.27.1" \
    "onnxruntime-gpu" \
    "psutil==5.9.8" \
    "aiohttp==3.9.5" \
    "rich>=13.9.4" \
    "colorama==0.4.6" \
    "cachetools>=5.3.0"



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
