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

WORKDIR /layoutlmv3
RUN python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

RUN cd / && git clone https://github.com/microsoft/unilm.git && cd unilm && git checkout 9102ed91f8e56baa31d7ae7e09e0ec98e77d779c
# Our requirements are installed in place of the layoutlmv3 requirements from unilm
COPY layoutlmv3/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
ADD metrics /metrics
RUN cd /metrics && pip3 install .
RUN cd /unilm/layoutlmv3 && pip3 install -e .