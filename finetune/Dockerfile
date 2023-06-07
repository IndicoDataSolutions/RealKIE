FROM tensorflow/tensorflow:2.11.0-gpu

# Note - this is not a complete finetune image and custom op build will fail
RUN apt-get update && apt install -y git
# TODO: switch this hash to development once that ben/fix_doc_labeler_label_text_assertions is merged.
RUN pip3 install git+https://github.com/IndicoDataSolutions/finetune.git@0df3f5533e2091118341917a6e013a25a7900e7a
ADD finetune/requirements.txt .
RUN pip3 install -r requirements.txt
ADD metrics /metrics
RUN cd /metrics && pip3 install .
ADD scripts/get_data.sh .
WORKDIR /finetune
