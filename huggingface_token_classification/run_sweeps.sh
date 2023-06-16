#!/bin/bash
./get_data.sh

for dataset in charities nda resource_contracts s1 fcc_invoices
do
    # Longformer low priority
    for hf_base_model in roberta-base microsoft/deberta-v3-base allenai/longformer-base-4096
    do
        python3 train_and_predict.py wandb_sweep $PROJECT $ENTITY $dataset $hf_base_model
    done
done
