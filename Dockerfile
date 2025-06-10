FROM python:3.11-slim

# Install system dependencies for OpenCV, TensorFlow, and pip builds
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

# Copy only requirements first
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy download script and run it
COPY download_yolo.py .
RUN python download_yolo.py

# Debug: List contents of /app/models and /app/yolo
RUN ls -lh /app/models || echo "No /app/models directory"
RUN ls -lh /app/yolo || echo "No /app/yolo directory"

# Copy the rest of the app code
COPY . .

# Expose port
EXPOSE 8080

# Launch with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
