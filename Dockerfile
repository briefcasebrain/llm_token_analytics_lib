# LLM Token Analytics Docker Image
# =================================

# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy library code
COPY llm_token_analytics/ ./llm_token_analytics/
COPY setup.py .
COPY README.md .

# Install the library
RUN pip install -e .

# Copy additional files
COPY api_server.py .
COPY config/ ./config/
COPY examples/ ./examples/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/cache /app/results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV HOST=0.0.0.0

# Expose port
EXPOSE 5000

# Default command (can be overridden)
CMD ["python", "api_server.py"]
