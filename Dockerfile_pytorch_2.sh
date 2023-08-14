# Adapted from Pytorch official docker file
# https://github.com/pytorch/pytorch/blob/main/Dockerfile

# Set up base Ubuntu 20 image
ARG BASE_IMAGE=ubuntu:20.04
ARG PYTHON_VERSION=3.10

FROM ${BASE_IMAGE} as dev-base
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

# Setup Conda, install Python and requirements.txt
FROM dev-base as conda
ARG PYTHON_VERSION=3.10
ARG TARGETPLATFORM # Automatically set by buildx
# Translate Docker's TARGETPLATFORM into miniconda arches
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  MINICONDA_ARCH=aarch64  ;; \
         *)              MINICONDA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -v -o ~/miniconda.sh -O  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh"
COPY ./requirements.txt .
# Manually invoke bash on miniconda script per https://github.com/conda/conda/issues/10431
RUN chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython && \
    /opt/conda/bin/python -mpip install -r requirements.txt && \
    /opt/conda/bin/conda clean -ya


FROM dev-base as submodule-update
WORKDIR /opt/pytorch
COPY . .
#RUN git submodule update --init --recursive

# Create a layer to store conda libraries
FROM conda as build
ARG CMAKE_VARS
WORKDIR /opt/pytorch
COPY --from=conda /opt/conda /opt/conda
#COPY --from=submodule-update /opt/pytorch /opt/pytorch
#RUN --mount=type=cache,target=/opt/ccache \
#    export eval ${CMAKE_VARS} && \
#    TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 8.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
#    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
#    python setup.py install


FROM conda as conda-installs
ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=118
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch-nightly
# Automatically set by buildx
RUN /opt/conda/bin/conda update -y conda
RUN /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -y python=${PYTHON_VERSION}
ARG TARGETPLATFORM
#RUN /opt/conda/bin/pip install torchelastic

FROM ${BASE_IMAGE} as official
ARG PYTORCH_VERSION
ARG TRITON_VERSION
ARG TARGETPLATFORM
ARG CUDA_VERSION
LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        && rm -rf /var/lib/apt/lists/*
COPY --from=conda-installs /opt/conda /opt/conda
RUN if test -n "${TRITON_VERSION}" -a "${TARGETPLATFORM}" != "linux/arm64"; then \
        DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends gcc; \
        rm -rf /var/lib/apt/lists/*; \
    fi
ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION ${PYTORCH_VERSION}
WORKDIR /workspace

FROM official as dev
# Should override the already installed version from the official-image stage
COPY --from=build /opt/conda /opt/conda

# Configure Jupyter

# Copy custom Jupyter config
COPY custom/ /root/.jupyter/

# Enable Jupyter widgets
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

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

# Install ML libs
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install bitsandbytes
RUN pip install bitsandbytes
RUN pip install safetensors
RUN pip install tokenizers
RUN pip install --upgrade --no-deps --force-reinstall -U huggingface_hub
RUN pip install --upgrade --no-deps --force-reinstall -U git+https://github.com/huggingface/transformers.git
RUN pip install  --upgrade --no-deps --force-reinstall -U git+https://github.com/huggingface/peft.git 
RUN pip install  --upgrade --no-deps --force-reinstall -U git+https://github.com/huggingface/accelerate.git

RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --sys-prefix

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



