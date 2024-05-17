# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV and clean up in one layer
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python dependencies file to the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . /app/

# Expose the port the app runs on
EXPOSE 5001

# Set environment variables for configuration
ENV DATABASE_URL=mongodb://mongo:27017/
ENV FLASK_APP app.py

# Health check to ensure the application is running
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD curl -f http://localhost:5001/ || exit 1

# Define command to start the app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]
