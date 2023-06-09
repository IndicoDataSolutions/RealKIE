#!/bin/bash
./get_data.sh

for dataset in charities nda resource_contracts s1
do
    for finetune_base_model in xdoc roberta-base
    do
        python3 train_and_predict.py wandb_sweep $PROJECT $ENTITY $dataset $finetune_base_model
    done
done