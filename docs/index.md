# RealKIE - Five Novel Datasets for Enterprise Key Information Extraction

## Accessing the data

Run `aws s3 sync s3://project-fruitfly <destination> --endpoint-url=https://s3.us-east-2.wasabisys.com --no-sign-request` to pull the data.

A backup copy of our datasets is available at https://zenodo.org/records/13327077 in case of issues with Wasabi.

## Running Baselines

Code is available on [github.com/IndicoDataSolutions/RealKIE](https://github.com/IndicoDataSolutions/RealKIE)

to run the baselines
```
docker-compose up -d
bash scripts/run_sweeps.sh
```
You will need to modify the variables at the top of the run_sweeps.sh script to point to the correct Weights and Biases entity and project.

## Visualization

To visualize the data
* First	download the data to <repo_path>/datasets/
* install streamlit `pip install streamlit`
* from the root of the repo run `streamlit run visualization/visualize_data.py`
