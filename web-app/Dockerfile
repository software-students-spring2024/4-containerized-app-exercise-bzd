# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable for database URL
ENV DATABASE_URL mongodb://mongo:27017/

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]
