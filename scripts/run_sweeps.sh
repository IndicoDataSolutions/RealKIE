PROJECT=dataset-paper-sweeps
ENTITY=indico
GPUS=(0 1 2)

cleanup() {
    echo "Caught SIGINT, killing child processes..."
    kill -- -$$
}

# Trap the SIGINT signal and execute the cleanup function
trap cleanup SIGINT

run_on_all_gpus() {
    local container="$1"
    shift
    local command="$@"
    local pids=()

    for gpu in "${GPUS[@]}"; do
        echo "Running on GPU ${gpu}"
        (docker-compose exec -T -e CUDA_VISIBLE_DEVICES="${gpu}" "${container}" ${command}) &
        pids+=($!)
        # Sleep for plenty of time to allow static files to download and avoic race consitions with wandb sweep initialization
        sleep 600
    done

    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
}

for dataset in charities nda resource_contracts s1
do
    for finetune_base_model in xdoc roberta-base roberta-large
    do
        run_on_all_gpus finetune python3 train_and_predict.py wandb_sweep $PROJECT $ENTITY $dataset $finetune_base_model
    done
    # Longformer low priority
    for hf_base_model in deberta-v3-base deberta-v3-large roberta-base roberta-large longformer-base-4096
    do
        run_on_all_gpus huggingface python3 train_and_predict.py wandb_sweep $PROJECT $ENTITY $dataset $hf_base_model
    done

    run_on_all_gpus layoutlmv3 python3 train_and_predict.py wandb_sweep $PROJECT $ENTITY $dataset
done

# Low priority.
for dataset in s1 resource_contracts
    for hf_base_model in Rakib/roberta-base-on-cuad
    do 
        run_on_all_gpus huggingface python3 train_and_predict.py wandb_sweep $PROJECT $ENTITY $dataset $hf_base_model
    done
done

# Any legal or financial models?