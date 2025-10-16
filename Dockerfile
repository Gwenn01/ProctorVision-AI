FROM python:3.10-bullseye

# --- Install system dependencies (for OpenCV, MediaPipe, TensorFlow, aiortc, PyAV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libatlas-base-dev \
    ffmpeg \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    libv4l-dev \
    libssl-dev \
    libffi-dev \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Work directory
WORKDIR /app

# --- Copy dependencies first for caching
COPY requirements.txt .

# --- Upgrade pip and install requirements
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --verbose -r requirements.txt

# --- Copy all source files
COPY . .

# --- Expose default HF Space port
EXPOSE 7860

# --- Start Flask app
CMD ["python", "app.py"]
