FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# System packages.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    tmux \
    ffmpeg \
    libsm6 \
    libxext6 \ 
    sudo 

# Download and install Micromamba.
RUN curl -sL https://micro.mamba.pm/api/micromamba/linux-64/1.1.0 \
  | sudo tar -xvj -C /usr/local bin/micromamba
ENV MAMBA_EXE=/usr/local/bin/micromamba \
    MAMBA_ROOT_PREFIX=/home/user/micromamba \
    CONDA_PREFIX=/home/user/micromamba \
    PATH=/home/user/micromamba/bin:$PATH

# Create a new environment and install Python.
RUN micromamba create -y -n base && \
    micromamba shell init --shell=bash --prefix="$MAMBA_ROOT_PREFIX"
RUN micromamba install python=3.11 pip -c conda-forge -y && python -m pip install --upgrade pip

ADD requirements.txt /deps/generative-models/requirements/requirements.txt
RUN python -m pip install -r /deps/generative-models/requirements/requirements.txt