ARG PYTORCH="1.12.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel


# Install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get update && apt-get install -y git apt-utils ninja-build cmake build-essential  \
    libopenblas-dev xterm xauth openssh-server tmux wget

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

##############################################
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
##############################################



# For faster build, use more jobs.
ENV MAX_JOBS=4
# ENV CUDA_HOME=/usr/local/cuda-12.1 
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
RUN cd MinkowskiEngine; python setup.py install --force_cuda --blas=openblas

RUN apt-get install libgl1-mesa-glx -y
RUN pip install spconv-cu113

RUN pip install nuscenes-devkit==1.1.9
RUN pip install tqdm pyquaternion

RUN pip install pytorch_lightning==1.6.0
RUN pip install pillow==8.3.1
RUN pip install scikit-image==0.18.2 
RUN pip install wandb==0.15.12
RUN pip install opencv-python==4.3.0.36  
#NOTE: Downgrading opencv-python from 4.4.10 to 4.3.0.36 solves both glib.so error on turing and seg fault due to wandb init 
## installing opencv-python-headless can also solve the issue of glib.so according to letian


WORKDIR /slidr
RUN git config --global --add safe.directory /slidr
RUN mkdir assets config datasets downstream model pretrain utils superpixels
#results weights output superpixels
