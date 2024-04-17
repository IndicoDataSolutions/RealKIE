import functools
import json
import logging
import os
import random
import tempfile
import traceback
from collections import defaultdict

import evaluate
import fire
import numpy as np
import pandas as pd
import wandb
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from datasets import Dataset
from metrics import metrics

logger = logging.getLogger(__name__)

TRAINING_LIB = "huggingface"


def overlaps(a, b):
    return a["start"] < b["end"] <= a["end"] or b["start"] < a["end"] <= b["end"]


def get_dataset(dataset_name, split, dataset_dir):
    csv = pd.read_csv(os.path.join(dataset_dir, dataset_name, f"{split}.csv"))
    data = []
    for row in csv.to_dict("records"):
        item = {
            "text": row["text"],
            "labels": json.loads(row["labels"]),
            "doc_path": row["document_path"],
        }
        data.append(item)
    return data


def clean_preds(preds, text, char_threshold=1):
    preds = sorted(preds, key=lambda x: x["start"])
    output = []
    for p in preds:
        if (
            output
            and p["start"] - output[-1]["end"] <= char_threshold
            and p["label"] == output[-1]["label"]
        ):
            output[-1]["end"] = p["end"]
        else:
            output.append(p)
    for p in output:
        p["text"] = text[p["start"] : p["end"]]
    return output


def undersample_empty_chunks(inputs, empty_chunk_ratio, no_label_idx=0, pad_idx=-100):
    keep_indices = []
    sample_indices = []
    for i, chunk_labels in enumerate(inputs["labels"]):
        is_empty = all(c in (no_label_idx, pad_idx) for c in chunk_labels)
        if is_empty:
            sample_indices.append(i)
        else:
            keep_indices.append(i)
    print("KEEPING", keep_indices)
    keep_indices = keep_indices + random.sample(
        sample_indices,
        min(len(sample_indices), int(empty_chunk_ratio * len(keep_indices))),
    )
    keep_indices = sorted(keep_indices)
    return {k: [v[i] for i in keep_indices] for k, v in inputs.items()}


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
    label_list = ["<NONE>"] + sorted(
        set(
            l["label"]
            for page in train_dataset
            for l in page["labels"]
            if l["label"] != "<NONE>"
        )
    )
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if "deberta" in base_model:
        # Deberta max length defaults to 1e32
        tokenizer.model_max_length = 512
    model = AutoModelForTokenClassification.from_pretrained(
        base_model, num_labels=num_labels, id2label=id_to_label, label2id=label_to_id
    )

    def get_hf_dataset(split, is_training=False):
        split_dataset = Dataset.from_generator(
            lambda: get_dataset(dataset_name, split, dataset_dir=dataset_dir)
        )
        tokenized = split_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=split_dataset.column_names,  # Drop all existing columns so that we can change the lengths of the output
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
        )
        if is_training and empty_chunk_ratio:
            tokenized = tokenized.map(
                functools.partial(
                    undersample_empty_chunks, empty_chunk_ratio=empty_chunk_ratio
                ),
                batched=True,
                remove_columns=tokenized.column_names,
                num_proc=preprocessing_num_workers,
                load_from_cache_file=not overwrite_cache,
                batch_size=1000,
            )
        return tokenized

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        labels = []
        doc_paths = []
        token_offsets = []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            offsets = tokenized_inputs.offset_mapping[batch_index]
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][
                batch_index
            ]
            label_ids = []
            doc_paths.append(examples["doc_path"][org_batch_index])
            token_offsets.append(
                [{"start": offset[0], "end": offset[1]} for offset in offsets]
            )
            assert len(word_ids) == len(offsets)
            for i, (word_idx, offset) in enumerate(zip(word_ids, offsets)):
                offset = {"start": offset[0], "end": offset[1]}
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                assert offset["end"] <= len(examples["text"][org_batch_index]), (
                    offset,
                    len(examples["text"][org_batch_index]),
                )
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_dicts = [
                        l
                        for l in examples["labels"][org_batch_index]
                        if overlaps(l, offset)
                    ]
                    for l in label_dicts:
                        l["matched"] = True
                    if label_dicts:
                        label = label_dicts[0]["label"]
                    else:
                        label = "<NONE>"
                    label_ids.append(label_to_id[label])

            labels.append(label_ids)
        for doc_labels in examples["labels"]:
            for l in doc_labels:
                if not l.get("matched", False):
                    print(f"Unmatched Label {l}")
        tokenized_inputs["labels"] = labels
        tokenized_inputs["doc_path"] = doc_paths
        tokenized_inputs["token_offsets"] = token_offsets
        return tokenized_inputs

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    train_dataset = get_hf_dataset("train", is_training=True)
    eval_dataset = get_hf_dataset("val")

    trainer = Trainer(
        model=model,
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
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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
            label_list=label_list,
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


def get_num_runs(sweep_id, entity, project):
    sweep = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}")
    if sweep.state not in {"RUNNING", "PENDING"}:
        print("This sweep is currently {}".format(sweep.state))
        exit(0)
    num_runs = len([None for r in sweep.runs if r.state in {"finished", "running"}])
    print(f"Current number of runs {num_runs}")
    return num_runs


def _run_agent(sweep_id, function, entity, project):
    while get_num_runs(sweep_id=sweep_id, entity=entity, project=project) <= 100:
        wandb.agent(sweep_id=sweep_id, function=function, entity=entity, project=project, count=1)
    print("This sweep is complete - exiting")
    exit(0)


def get_matching_sweep(project, entity, sweep_id_config):
    project = wandb.Api().project(project, entity=entity)
    try:
        sweeps = project.sweeps()
    except wandb.errors.CommError:
        return None
    for s in sweeps:
        sweep_config = s.config["parameters"]
        if sweep_id_config.items() <= sweep_config.items():
            return s.id
    return None


def setup_and_run_sweep(
    project, entity, dataset_name, base_model, dataset_dir="/datasets",
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
    
    sweep_configs = {
        "method": "bayes",
        "metric": {"name": "val_macro_f1", "goal": "maximize"},
        "parameters": {
            # Add these values in here to make resuming easier.
            **sweep_id_config,
            # Sweep Params
            "empty_chunk_ratio": {
                "distribution": "log_uniform_values",
                "min": 1e-2,
                "max": 1000.0,
            },
            "num_train_epochs": {"distribution": "int_uniform", "min": 1, "max": 16},
            "per_device_train_batch_size": {
                "distribution": "int_uniform",
                "min": 1,
                "max": 4,
            },
            "per_device_eval_batch_size": {
                "value": 2,
            },
            "gradient_accumulation_steps": {
                "distribution": "int_uniform",
                "min": 1,
                "max": 8,
            },
            "warmup_ratio": {"distribution": "uniform", "min": 0, "max": 0.5},
            "warmup_steps": {"value": 0},  # To ensure precedence goes to warmup_ratio.
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-8,
                "max": 1e-2,
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1.0,
            },
            "max_grad_norm": {
                "distribution": "log_uniform_values",
                "min": 1e-3,
                "max": 1e5,
            },
            "lr_scheduler_type": {
                "values": [
                    "linear",
                    "cosine",
                    "cosine_with_restarts",
                    "constant",
                    "constant_with_warmup",
                    "inverse_sqrt",
                ]
            },
        },
    }
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
