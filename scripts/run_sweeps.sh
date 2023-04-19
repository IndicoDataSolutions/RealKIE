PROJECT=test_finetune_project_3
ENTITY=benleetownsend

for dataset in fcc_invoices
do
    for finetune_base_model in layoutlm xdoc bert-base roberta-base bert-large roberta-large
    do
        docker-compose exec finetune python3 train_and_predict.py wandb_sweep $PROJECT $ENTITY $dataset $finetune_base_model
    done

    for hf_base_model in distilbert-base-uncased # longformer-base-4096 deberta-v3-base deberta-v3-large
    do
        docker-compose exec huggingface python3 train_and_predict.py wandb_sweep $PROJECT $ENTITY $dataset $hf_base_model
    done
    docker-compose exec layoutlmv3 python3 train_and_predict.py wandb_sweep $PROJECT $ENTITY $dataset
done