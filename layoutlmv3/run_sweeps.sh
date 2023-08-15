#!/bin/bash
./get_data.sh

for dataset in s1_truncated # resource_contracts fcc_invoices # charities nda 
do
    python3 train_and_predict.py wandb_sweep $PROJECT $ENTITY $dataset
done