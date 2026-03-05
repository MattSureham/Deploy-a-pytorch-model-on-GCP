# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model directory
RUN mkdir -p model

# Download and save a pre-trained model if none exists
# (This will be overridden by mounted model volume in production)
RUN python -c "
import torch
from model import MNISTClassifier
model = MNISTClassifier()
torch.save({
    'model_state_dict': model.state_dict(),
    'accuracy': 0.0
}, 'model/mnist_model.pth')
print('Initialized empty model')
"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=model/mnist_model.pth
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "main.py"]
