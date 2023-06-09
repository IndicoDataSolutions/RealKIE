FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

RUN apt-get update && \
    apt-get install -y libglib2.0-0 wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y software-properties-common gcc git && \
    add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.8 python3-distutils python3-pip python3-apt

RUN pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /scripts
ADD huggingface_token_classification/requirements.txt /scripts
ADD metrics /metrics
RUN cd /metrics && pip3 install .
RUN pip3 install -r /scripts/requirements.txt
ADD huggingface_token_classification /scripts
ADD scripts/get_data.sh .