import json
import os

import fire
import pandas as pd
import wandb

from utils.helpers import (
    get_matching_sweep,
    get_dataset,
    get_model_input_from_csv,
    get_dataset_path,
    run_agent,
)
from finetune_helpers import get_base_model
from all_sweep_configs import get_configs_for_model


TRAINING_LIB = "finetune"


def run_predictions(model, dataset_name, dataset_dir, split, output_path, is_document):
    test_data = pd.read_csv(get_dataset_path(dataset_name, dataset_dir, split))
    preds = model.predict(
        get_model_input_from_csv(
            test_data, is_document=is_document, dataset_dir=dataset_dir
        )
    )
    test_data["preds"] = [json.dumps(p) for p in preds]
    test_data.to_csv(output_path)


def train_and_predict(
    dataset_name,
    dataset_dir="/datasets",
    base_model="layoutlm",
    output_dir=None,
    **model_config,
):
    if output_dir is None:
        output_dir = os.path.join("outputs", dataset_name, "model_output")
    model_def = get_base_model(base_model)
    model = model_def["model_type"](base_model=model_def["base_model"], **model_config)
    model.fit(
        *get_dataset(
            dataset_name, dataset_dir, "train", is_document=model_def["is_document"]
        )
    )
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        run_predictions(
            model=model,
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            split=split,
            output_path=os.path.join(output_dir, f"{split}_predictions.csv"),
            is_document=model_def["is_document"],
        )


def setup_and_run_sweep(
    project, entity, dataset_name, base_model, dataset_dir="/datasets"
):
    sweep_id_config = {
        "dataset_name": {"value": dataset_name},
        "dataset_dir": {"value": dataset_dir},
        "base_model": {"value": base_model},
        "training_lib": {"value": TRAINING_LIB},
    }
    sweep_id = get_matching_sweep(project, entity, sweep_id_config)
    if sweep_id is not None:
        print(f"Resuming Sweep with ID: {sweep_id}")
        return run_agent(
            sweep_id,
            entity=entity,
            project=project,
            training_lib=TRAINING_LIB,
            train_and_predict=train_and_predict,
        )
    sweep_configs = get_configs_for_model(TRAINING_LIB, sweep_id_config)
    sweep_id = wandb.sweep(sweep_configs, project=project, entity=entity)
    print(f"Your sweep id is {sweep_id}")
    return run_agent(
        sweep_id,
        entity=entity,
        project=project,
        training_lib=TRAINING_LIB,
        train_and_predict=train_and_predict,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "train_and_predict": train_and_predict,
            "wandb_sweep": setup_and_run_sweep,
            "wandb_resume_sweep": run_agent,
        }
    )
