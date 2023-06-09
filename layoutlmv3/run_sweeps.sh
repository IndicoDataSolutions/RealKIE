#!/bin/bash
./get_data.sh

for dataset in charities nda resource_contracts s1
do
    python3 train_and_predict.py wandb_sweep $PROJECT $ENTITY $dataset
done