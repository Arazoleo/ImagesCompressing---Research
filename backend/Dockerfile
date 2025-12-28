# Use Python 3.9 with Ubuntu for better OpenGL support
FROM python:3.9

# Set working directory
WORKDIR /app

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads processed temp logs

# Expose port
EXPOSE 8001

# Run the application
CMD ["python", "main.py"]
