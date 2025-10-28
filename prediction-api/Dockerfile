FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY earthquake_tsunami_predictor.py .
COPY model.pkl .

# Set environment variable for Flask
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Run Flask directly (simple and works!)
CMD python app.py