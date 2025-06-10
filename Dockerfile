# Use a compatible and slim base image
FROM python:3.11-slim

# Install system dependencies required for OpenCV, TensorFlow, and pip builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only the requirements first
COPY requirements.txt .

# Upgrade pip and install dependencies line-by-line for debugging
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies one at a time to identify failures
RUN cat requirements.txt | xargs -n 1 pip install --no-cache-dir

# Copy the full app code
COPY . .

# Download required files (like YOLO models)
RUN python download_yolo.py

# Expose the Flask/Gunicorn port
EXPOSE 8080

# Launch the app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
