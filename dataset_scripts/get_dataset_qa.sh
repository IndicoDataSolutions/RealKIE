CUDA_VISIBLE_DEVICES=1 python3 dataset_scripts/dataset_qa.py --dataset_name="fcc_invoices" &
#CUDA_VISIBLE_DEVICES=2 python3 dataset_scripts/dataset_qa.py --dataset_name="charities_v2" &
#CUDA_VISIBLE_DEVICES=3 python3 dataset_scripts/dataset_qa.py --dataset_name="s1_v2" &
#CUDA_VISIBLE_DEVICES=1 python3 dataset_scripts/dataset_qa.py --dataset_name="resource_contracts_v2" &
#CUDA_VISIBLE_DEVICES=3 python3 dataset_scripts/dataset_qa.py --dataset_name="nda_v2" &
wait
