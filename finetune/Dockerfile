FROM tensorflow/tensorflow:2.11.0-gpu

# Note - this is not a complete finetune image and custom op build will fail
RUN apt-get update && apt install -y git
RUN pip3 install git+https://github.com/IndicoDataSolutions/finetune.git@13e5ffab65b08f91e297ab1dac2977a997bec5a5
ADD finetune/requirements.txt .
RUN pip3 install -r requirements.txt
ADD metrics /metrics
RUN cd /metrics && pip3 install .
WORKDIR /finetune
