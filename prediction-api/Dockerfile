FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including gcloud SDK tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Google Cloud Storage library
RUN pip install --no-cache-dir google-cloud-storage

# Copy application code (NOT model.pkl)
COPY app.py .
COPY earthquake_tsunami_predictor.py .

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV MODEL_BUCKET=models_tsunami_2023019
ENV MODEL_FILE=deployed_model.pkl

# Expose the port
EXPOSE 8080

# Run Flask directly
CMD python app.py