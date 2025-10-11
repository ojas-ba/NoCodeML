# Builder stage
FROM python:3.11-slim as builder

WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
# libgomp1 is required for LightGBM and other ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /usr/src/app/wheels /wheels

# Install Python packages
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Copy application code
COPY ./app ./app

# Copy Alembic configuration and migrations
COPY alembic.ini .
COPY alembic ./alembic

# Create necessary directories for data storage
RUN mkdir -p /app/datasets /app/models

# Set Python path
ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]