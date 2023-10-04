import json
import os
import tempfile
import traceback

import fire
import pandas as pd
import wandb

import tensorflow as tf

from helpers import (
    get_matching_sweep,
    get_num_runs,
    get_dataset,
    get_model_input_from_csv,
    get_dataset_path,
)
from finetune_helpers import get_base_model
from all_sweep_configs import get_configs_for_model

import metrics


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


def get_sweep_config():
    config = dict(**wandb.config)
    tl = config.pop("training_lib")
    assert tl == TRAINING_LIB, f"You are trying to resume a {tl} using {TRAINING_LIB}"
    return config


def run_agent(sweep_id, entity, project):
    def train_model():
        try:
            wandb.init(save_code=True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                train_and_predict(output_dir=tmp_dir, **get_sweep_config())
                for split in ["val", "test"]:
                    split_metrics = metrics.get_metrics_dict(
                        os.path.join(tmp_dir, f"{split}_predictions.csv"), split=split
                    )
                    wandb.log(split_metrics)
        except:
            # Seems like this method of running with wandb swallows the tracebacks.
            print(traceback.format_exc())
            raise
        finally:
            tf.compat.v1.reset_default_graph()

    _run_agent(sweep_id=sweep_id, function=train_model, entity=entity, project=project)


def _run_agent(sweep_id, function, entity, project):
    while get_num_runs(sweep_id=sweep_id, entity=entity, project=project) <= 100:
        wandb.agent(
            sweep_id=sweep_id,
            function=function,
            entity=entity,
            project=project,
            count=1,
        )
    print("This sweep is complete - exiting")
    exit(0)


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
        return run_agent(sweep_id, entity=entity, project=project)
    sweep_configs = get_configs_for_model("finetune", sweep_id_config)
    sweep_id = wandb.sweep(sweep_configs, project=project, entity=entity)
    print(f"Your sweep id is {sweep_id}")
    return run_agent(sweep_id, entity=entity, project=project)


if __name__ == "__main__":
    fire.Fire(
        {
            "train_and_predict": train_and_predict,
            "wandb_sweep": setup_and_run_sweep,
            "wandb_resume_sweep": run_agent,
        }
    )
