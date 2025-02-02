# an official PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV WORKDIR=/app
ENV VENV_PATH=$WORKDIR/venv

WORKDIR $WORKDIR

RUN apt-get update && apt-get install -y \
    python3-venv \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . $WORKDIR

RUN python3 -m venv $VENV_PATH

ENV PATH="$VENV_PATH/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]
