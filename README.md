# RealKIE - Five Novel Datasets for Enterprise Key Information Extraction

## Accessing the data

Run `aws s3 sync s3://project-fruitfly <destination> --endpoint-url=https://s3.us-east-2.wasabisys.com --no-sign-request` to pull the data.

## Running Baselines

to run the baselines
```
bash scripts/get_data.sh
docker compose up -d
bash scripts/run_sweeps.sh
```
You will need to modify the variables at the top of the run_sweeps.sh script to point to the correct Weights and Biases entity and project.
The results will be available on Weights and Biases. Some scripts for analysis can be found in `results_analysis_scripts/`

