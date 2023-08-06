FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04 AS base-image

# Upgrade and install system libraries
RUN apt-get -y update \
    && ACCEPT_EULA=Y apt-get upgrade -qq \
    && apt-get -y install \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libpq-dev -y

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r requirements.txt
