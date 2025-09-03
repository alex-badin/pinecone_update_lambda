FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage (includes all dependencies)
FROM base as development
COPY . .
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
CMD ["python", "main.py"]

# Production stage (optimized for AWS)
FROM base as production

# Copy source code
COPY . .

# Create directories for persistent storage mount points
RUN mkdir -p /data/databases /data/logs /data/cache /data/exports

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run the application
CMD ["python", "main.py"]
