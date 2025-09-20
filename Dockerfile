FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

# Install Python, NVIDIA utilities, and other dependencies
RUN apt-get update && apt-get install -y python3 python3-pip bash-completion git make build-essential gcc ffmpeg cuda-nvcc-13-0 libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /self-forcing/requirements.txt
WORKDIR /self-forcing

ENV LD_LIBRARY_PATH=/usr/local/cuda-13/lib64/

RUN pip install --break -r requirements.txt
RUN pip install --break flash-attn --no-build-isolation && \
    rm -rf /self-forcing

# Run with --gpus all and mount Self-Forcing to /self-forcing