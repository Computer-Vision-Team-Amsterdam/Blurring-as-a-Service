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

WORKDIR /opt/app

RUN pip install --no-cache-dir --upgrade pip

RUN curl -sSL https://install.python-poetry.org | python3 -
RUN /root/.local/bin/poetry config virtualenvs.create false

COPY pyproject.toml .
COPY poetry.lock .

ENV PATH="/opt/venv/bin:$PATH"

RUN /root/.local/bin/poetry update --no-ansi --no-interaction
RUN /root/.local/bin/poetry install --no-ansi --no-interaction --no-root