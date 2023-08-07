PROJECT=dataset-paper-sweeps
ENTITY=indico
GPUS=(0 1 2)

cleanup() {
    echo "Caught SIGINT, killing child processes..."
    kill -- -$$
}

# Trap the SIGINT signal and execute the cleanup function
trap cleanup SIGINT

run_on_all_gpus_old() {
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

run_on_all_gpus() {
    local container="$1"
    shift
    local command="$@"
    local pids=()

    local isFirstIteration=true

    for gpu in "${GPUS[@]}"; do
        echo "Running on GPU ${gpu}"
        (docker-compose exec -T -e CUDA_VISIBLE_DEVICES="${gpu}" "${container}" ${command}) &
        local cmd_pid=$!
        pids+=($cmd_pid)
        # Sleep for plenty of time to allow static files to download and avoid race conditions with wandb sweep initialization
	# Exit early if the first command finishes.
        if $isFirstIteration; then
            for (( i=0; i<600; i++ )); do
                # Check if the command has finished
                if ! kill -0 $cmd_pid 2> /dev/null; then
                    break
                fi
                sleep 1
            done
            isFirstIteration=false
        fi
    done

    for pid in "${pids[@]}"; do
        if [ $pid -ne $cmd_pid ]; then
            wait "${pid}"
        fi
    done
}

for dataset in resource_contracts s1_truncated fcc_invoices # charities nda
do
    for finetune_base_model in xdoc roberta-base 
    # roberta-large
    do
        run_on_all_gpus finetune python3 train_and_predict.py wandb_sweep $PROJECT $ENTITY $dataset $finetune_base_model
    done
    # Longformer low priority
    for hf_base_model in roberta-base microsoft/deberta-v3-base allenai/longformer-base-4096
    # deberta-v3-large roberta-large
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
