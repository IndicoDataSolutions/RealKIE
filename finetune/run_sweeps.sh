#!/bin/bash
PROJECT=dataset-paper-sweeps-k8s-test
ENTITY=indico

for dataset in nda
do
    for finetune_base_model in xdoc roberta-base
    do
        python3 train_and_predict.py wandb_sweep $PROJECT $ENTITY $dataset $finetune_base_model
    done
done