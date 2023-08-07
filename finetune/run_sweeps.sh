#!/bin/bash
./get_data.sh

for dataset in s1_truncated fcc_invoices # charities nda resource_contracts
do
    for finetune_base_model in xdoc roberta-base
    do
        python3 train_and_predict.py wandb_sweep $PROJECT $ENTITY $dataset $finetune_base_model
    done
done