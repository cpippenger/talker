# Adapted from Pytorch official docker file
# https://github.com/pytorch/pytorch/blob/main/Dockerfile

# Set up base Ubuntu 20 image
#ARG BASE_IMAGE=ubuntu:20.04
ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ARG PYTHON_VERSION=3.10

FROM ${BASE_IMAGE}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        gcc \
        libjpeg-dev \
        libpng-dev \
        nano \
        curl \
        unzip \
        wget &&\
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update

RUN apt-get install -y python3.10

RUN apt-get install -y python3-pip

RUN apt-get update

# Configure Jupyter
RUN pip3 install --upgrade notebook==6.1.5

# Copy custom Jupyter config
COPY custom/ /root/.jupyter/

# Enable Jupyter widgets
#RUN pip3 install jupyter_contrib_nbextensions
#RUN jupyter contrib nbextension install --user
#RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

# TODO: Generate notebook config

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        gcc \
        nano \
        curl \
        unzip \
        wget

COPY requirements.txt requirements.txt

# Install ML libs
RUN pip3 install -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#RUN pip3 install deepspeed
RUN pip install bitsandbytes
RUN pip install safetensors
RUN pip install tokenizers
RUN pip install --upgrade --no-deps --force-reinstall -U huggingface_hub
RUN pip install --upgrade --no-deps --force-reinstall -U git+https://github.com/huggingface/transformers.git
RUN pip install  --upgrade --no-deps --force-reinstall -U git+https://github.com/huggingface/peft.git 
RUN pip install  --upgrade --no-deps --force-reinstall -U git+https://github.com/huggingface/accelerate.git

#RUN pip install jupyter_contrib_nbextensions
#RUN jupyter contrib nbextension install --sys-prefix

# Set up aws
# Source: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
#RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
#RUN unzip awscliv2.zip
#RUN ./aws/install

EXPOSE 3141
EXPOSE 8888
EXPOSE 6901
EXPOSE 5901
EXPOSE 5900
EXPOSE 22

RUN apt-get update

# Run Jupyter widgets
#RUN jupyter notebook --ip=0.0.0.0 --allow-root --no-browser &



