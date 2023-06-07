api_token_path="/home/bentownsend/indico_api_token.txt"
update=False

python3.7 dataset_scripts/get_datasets.py --name="fcc_invoices" --dataset_id=12716 --labelset_id=27178 --label_col="FCC Invoices - Dataset Paper" --text_col=document --host=app.indico.io --api_token_path=$api_token_path --update=$update &
#python3.7 dataset_scripts/get_datasets.py --name="charities_v2" --dataset_id=11719 --labelset_id=25133 --label_col="Charities - Datasets Paper - V7" --text_col=file --host=app.indico.io --api_token_path=$api_token_path  --update=$update &
#python3.7 dataset_scripts/get_datasets.py --name="s1_v2" --dataset_id=11723 --labelset_id=25159 --label_col="S1 extraction" --text_col=document --host=app.indico.io --api_token_path=$api_token_path --update=$update &
#python3.7 dataset_scripts/get_datasets.py --name="resource_contracts_v2" --dataset_id=11674 --labelset_id=25117 --label_col="resource contracts extraction" --text_col=document --host=app.indico.io --api_token_path=$api_token_path --update=$update &
#python3.7 dataset_scripts/get_datasets.py --name="nda_v2" --dataset_id=9692 --labelset_id=21979 --label_col=question --text_col=document --host=app.indico.io --api_token_path=$api_token_path --update=$update &
wait