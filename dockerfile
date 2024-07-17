# Use an official Python runtime as a parent image 
FROM python:3.10-slim

# Set environment variables 
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1

# Set the working directory in the container 
WORKDIR /app

# Install git using apt cache
RUN apt-get update && apt-get install -y git && apt-get clean

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies using pip cache
RUN pip install -r requirements.txt

# Install lida from the source in editable mode using pip cache
RUN pip install -e .

# Expose the port that the application will listen on 
EXPOSE 8080

# Start the Web UI
Entrypoint lida ui --host 0.0.0.0 --port 8080 --docs