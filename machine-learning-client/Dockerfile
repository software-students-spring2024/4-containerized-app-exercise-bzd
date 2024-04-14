# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies for OpenCV and general operations
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python dependencies file to the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . /app/

# If your application serves web traffic on port 80, expose this port. Otherwise, this line can be omitted.
EXPOSE 80

# It's better to set environment variables like DATABASE_URL in the docker-compose file or through the Docker run command for security reasons.
ENV DATABASE_URL=mongodb://mongo:27017/

# Command to run the application
CMD ["python", "camera.py"]
