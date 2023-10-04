import functools
import json
import logging
import os
import tempfile
import traceback
from collections import defaultdict

import evaluate
import fire
import numpy as np
import pandas as pd
import wandb
from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from metrics import metrics

from hf_helpers import HFLoader
from helpers import (
    get_matching_sweep,
    get_num_runs,
    clean_preds,
    get_dataset,
)
from all_sweep_configs import get_configs_for_model

logger = logging.getLogger(__name__)

TRAINING_LIB = "huggingface"


def train_and_predict(
    dataset_name,
    dataset_dir="/datasets",
    base_model="distilbert-base-uncased",
    output_dir=None,
    preprocessing_num_workers=None,
    overwrite_cache=True,
    empty_chunk_ratio=2.0,
    per_device_train_batch_size=2,
    num_train_epochs=16,
    gradient_accumulation_steps=1,
    **training_args,
):
    if output_dir is None:
        output_dir = os.path.join("outputs", dataset_name, "model_output")

    train_dataset = get_dataset(dataset_name, "train", dataset_dir)

    loader = HFLoader(
        dataset_name,
        dataset_dir,
        base_model,
        preprocessing_num_workers,
        overwrite_cache,
        empty_chunk_ratio,
    )
    train_dataset = loader.get_hf_dataset("train", is_training=True)
    eval_dataset = loader.get_hf_dataset("val")
    data_collator = DataCollatorForTokenClassification(
        tokenizer=loader.tokenize_and_align_labels
    )

    trainer = Trainer(
        model=loader.model,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            save_total_limit=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **training_args,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=loader.tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: loader.compute_metrics(p),
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Prediction
    logger.info("*** Predict ***")

    def run_predictions(trainer, split, label_list, output_path):
        hf_dataset = get_hf_dataset(split)
        orig_dataset = get_dataset(dataset_name, split, dataset_dir)

        predictions, _, _ = trainer.predict(hf_dataset)
        predictions = np.argmax(predictions, axis=2)
        pred_by_path = defaultdict(list)
        for prediction, example in zip(predictions, hf_dataset):
            preds = []
            for pred, offsets in zip(prediction, example["token_offsets"]):
                pred = label_list[pred]
                if pred != "<NONE>":
                    preds.append(
                        {
                            "label": pred,
                            "start": offsets["start"],
                            "end": offsets["end"],
                        }
                    )
            pred_by_path[example["doc_path"]] += preds
        pred_records = []
        for row in orig_dataset:
            pred_records.append(
                {
                    "doc_path": row["doc_path"],
                    "preds": json.dumps(
                        clean_preds(pred_by_path[row["doc_path"]], row["text"])
                    ),
                    "labels": json.dumps(row["labels"]),
                    "text": row["text"],
                }
            )
        pd.DataFrame.from_records(pred_records).to_csv(output_path)

    for split in ["train", "val", "test"]:
        run_predictions(
            trainer,
            split,
            label_list=loader.label_list,
            output_path=os.path.join(output_dir, f"{split}_predictions.csv"),
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
    project,
    entity,
    dataset_name,
    base_model,
    dataset_dir="/datasets",
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
    sweep_configs = get_configs_for_model("hf", sweep_id_config)
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
