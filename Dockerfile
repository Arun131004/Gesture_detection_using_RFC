# Use Python 3.8 image
FROM python:3.8.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages first
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data

# Set environment variable for port
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Start the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app --timeout 300
