# Start from an image with ubuntu 16.04, python 3.5 and tensorflow-gpu 1.12.0
FROM tensorflow/tensorflow:1.12.0-gpu-py3

# Install Python packages
COPY requirements.txt /
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

# Install dependencies for OpenCV and SocketIO
RUN apt-get update && apt-get install -y \
    netbase \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    python3-tk
