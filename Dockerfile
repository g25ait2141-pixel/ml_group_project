# Use lightweight Python image
FROM python:3.11-slim

# Prevent Python output buffering
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src

# Default command to run inference
CMD ["python", "src/inference.py"]
