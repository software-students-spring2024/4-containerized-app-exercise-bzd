# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies for OpenCV and general operations
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \  
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python dependencies file to the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . /app/

# Expose the port the app runs on if necessary
EXPOSE 5000

# Define environment variables if necessary
ENV DATABASE_URL mongodb://mongo:27017/

# Command to run the application
CMD ["python", "camera.py"]
