FROM mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04 AS base-image

# Upgrade and install system libraries
RUN apt-get -y update \
    && ACCEPT_EULA=Y apt-get upgrade -qq \
    && apt-get -y install \
        build-essential \
        curl \
        ffmpeg \
        libsm6 \
        libxext6 \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

RUN pip install uv

COPY pyproject.toml .

RUN uv pip install --system -r pyproject.toml --prerelease=allow 
