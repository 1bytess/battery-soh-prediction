# api/Dockerfile (Production with TensorFlow)

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for TensorFlow
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_prod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_prod.txt

# Copy application files
COPY app.py .

# Copy your model directory (note: using 'model' not 'models')
COPY model/ model/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV DEBUG=False
ENV TF_CPP_MIN_LOG_LEVEL=2

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run with gunicorn for production (with more workers for TensorFlow)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--worker-class", "sync", "--timeout", "120", "--max-requests", "1000", "--max-requests-jitter", "50", "app:app"]