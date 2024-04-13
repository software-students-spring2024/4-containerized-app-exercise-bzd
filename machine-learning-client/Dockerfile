# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app/machine-learning-client

COPY requirements.txt ./

# OpenCV prerequisites
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /usr/src/app


# Install Python dependencies
RUN pip install --no-cache-dir torch torchvision opencv-python-headless

# Make port 80 available to the world outside this container
EXPOSE 80



# Run app.py when the container launches
CMD ["python", "camera.py","python3"]
