import json
import os
import tempfile
import traceback

import fire
import pandas as pd
import wandb

from finetune import DocumentLabeler, SequenceLabeler
from finetune.base_models import (
    BERTLarge,
    BERTModelCased,
    LayoutLM,
    RoBERTa,
    ROBERTALarge,
    XDocBase,
)
from metrics import metrics

TRAINING_LIB = "finetune"


def get_dataset_path(dataset_name, dataset_dir, split):
    return os.path.join(dataset_dir, dataset_name, f"{split}.csv")


def get_model_input_from_csv(csv, is_document):
    if is_document:
        return [json.loads(o) for o in csv.ocr]
    return csv.text


def get_dataset(dataset_name, dataset_dir, split, is_document):
    csv = pd.read_csv(get_dataset_path(dataset_name, dataset_dir, split))
    labels = [json.loads(l) for l in csv.labels]
    return get_model_input_from_csv(csv, is_document=is_document), labels


def get_base_model(base_model):
    base_model = base_model.lower()
    return {
        "layoutlm": {
            "base_model": LayoutLM,
            "model_type": DocumentLabeler,
            "is_document": True,
        },
        "xdoc": {
            "base_model": XDocBase,
            "model_type": DocumentLabeler,
            "is_document": True,
        },
        "bert-base": {
            "base_model": BERTModelCased,
            "model_type": SequenceLabeler,
            "is_document": False,
        },
        "roberta-base": {
            "base_model": RoBERTa,
            "model_type": SequenceLabeler,
            "is_document": False,
        },
        "bert-large": {
            "base_model": BERTLarge,
            "model_type": SequenceLabeler,
            "is_document": False,
        },
        "roberta-large": {
            "base_model": ROBERTALarge,
            "model_type": SequenceLabeler,
            "is_document": False,
        },
    }[base_model]


def run_predictions(model, dataset_name, dataset_dir, split, output_path, is_document):
    test_data = pd.read_csv(get_dataset_path(dataset_name, dataset_dir, split))
    preds = model.predict(get_model_input_from_csv(test_data, is_document=is_document))
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
    model = model_def["model_type"](base_model=model_def["base_model"], **model_config,)
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


def run_agent(sweep_id):
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

    wandb.agent(sweep_id=sweep_id, function=train_model)


def setup_and_run_sweep(
    project, entity, dataset_name, base_model, dataset_dir="/datasets",
):
    sweep_configs = {
        "method": "bayes",
        "metric": {"name": "val_macro_f1", "goal": "minimize"},
        "parameters": {
            # Add these values in here to make resuming easier.
            "dataset_name": {"value": dataset_name},
            "dataset_dir": {"value": dataset_dir},
            "base_model": {"value": base_model},
            "training_lib": {"value": TRAINING_LIB},
            # Sweep Params
            "auto_negative_sampling": {"values": [False, True]},
            "max_empty_chunk_ratio": {
                "distribution": "log_uniform_values",
                "min": 1e-2,
                "max": 1000.0,
            },
            "lr": {"distribution": "log_uniform_values", "min": 1e-8, "max": 1e-2},
            "batch_size": {"distribution": "int_uniform", "min": 1, "max": 8},
            "n_epochs": {"distribution": "int_uniform", "min": 1, "max": 128},
            "class_weights": {"values": [None, "linear", "sqrt", "log"]},
            "lr_warmup": {"distribution": "uniform", "min": 0, "max": 0.5},
            "collapse_whitespace": {"values": [True, False]},
            "max_grad_norm": {
                "distribution": "log_uniform_values",
                "min": 1e-3,
                "max": 1e5,
            },
            "l2_reg": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1.0},
        },
    }
    sweep_id = wandb.sweep(sweep_configs, project=project, entity=entity)
    print(f"Your sweep id is {sweep_id}")
    run_agent(sweep_id)


if __name__ == "__main__":
    fire.Fire(
        {
            "train_and_predict": train_and_predict,
            "wandb_sweep": setup_and_run_sweep,
            "wandb_resume_sweep": run_agent,
        }
    )
