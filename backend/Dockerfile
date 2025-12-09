FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy backend dependency metadata first (for layer caching)
# We keep backend as a proper Python package under /app/backend
COPY backend/pyproject.toml backend/README.md /app/backend/

# Install backend dependencies (use uv inside Docker)
WORKDIR /app/backend
RUN uv pip install --system --no-cache -e .

# Copy application code (entire backend package)
WORKDIR /app
COPY backend /app/backend

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application using the backend package path
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
