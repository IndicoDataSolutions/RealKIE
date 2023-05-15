# Based on https://github.com/microsoft/unilm/blob/master/layoutlmv3/examples/run_funsd_cord.py
import functools
import json
import logging
import os
import random
import tempfile
import traceback
from collections import defaultdict
import gzip

import fire
import numpy as np
import pandas as pd
import torch
import wandb
from layoutlmft.data import DataCollatorForKeyValueExtraction
from layoutlmft.data.image_utils import (
    Compose,
    RandomResizedCropAndInterpolationWithTwoPic,
    pil_loader,
)
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from torchvision import transforms
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.utils import check_min_version

from datasets import Dataset, load_metric
from metrics import metrics

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)

TRAINING_LIB = "unilm/layoutlmv3"


def overlaps(a, b):
    return a["start"] < b["end"] <= a["end"] or b["start"] < a["end"] <= b["end"]


def fix_start_end(l, doc_offset):
    l = dict(l)
    l["start"] = max(l["start"] - doc_offset["start"], 0)
    l["end"] = min(
        l["end"] - doc_offset["start"], doc_offset["end"] - doc_offset["start"]
    )
    return l


def join_positions(positions):
    return {
        "left": min(p["left"] for p in positions),
        "right": max(p["right"] for p in positions),
        "top": min(p["top"] for p in positions),
        "bottom": max(p["bottom"] for p in positions),
    }


def add_line_bboxes(page_text, page_tokens):
    lines = page_text.split("\n")
    line_spans = []
    for line in lines:
        start = line_spans[-1]["end"] + 1 if line_spans else 0
        line_spans.append({"start": start, "end": start + len(line)})
        line_tokens = [
            tok for tok in page_tokens if overlaps(tok["page_offset"], line_spans[-1])
        ]
        if len(line_tokens) == 0:
            continue
        line_spans[-1]["position"] = join_positions(
            [tok["position"] for tok in line_tokens]
        )
        for tok in line_tokens:
            tok["line_position"] = line_spans[-1]["position"]


def get_dataset(dataset_name, split, dataset_dir):
    csv = pd.read_csv(os.path.join(dataset_dir, dataset_name, f"{split}.csv"))
    data = []
    for row in csv.to_dict("records"):
        labels = json.loads(row["labels"])
        ocr = json.loads()
        with gzip.open(os.path.join(dataset_dir, row["ocr"]), 'rt') as fp:
            ocr = json.loads(fp.read())
        images = json.loads(row["image_files"])
        for page_ocr, page_image in zip(ocr, images):
            doc_offset = page_ocr["pages"][0]["doc_offset"]
            page_labels = [l for l in labels if overlaps(l, doc_offset)]
            add_line_bboxes(page_ocr["pages"][0]["text"], page_ocr["tokens"])
            item = {
                "text": page_ocr["pages"][0]["text"],
                "labels": page_labels,
                "page_tokens": page_ocr["tokens"],
                "page_image": os.path.join(dataset_dir, page_image),
                "page_size": page_ocr["pages"][0]["size"],
                "doc_path": row["document_path"],
                "page_num": page_ocr["pages"][0]["page_num"],
                "page_offset": doc_offset["start"],
            }
            data.append(item)
    return data


def get_layoutlm_position(position, page_size):
    return [
        int(1000 * position["left"] / page_size["width"]),
        int(1000 * position["top"] / page_size["height"]),
        int(1000 * position["right"] / page_size["width"]),
        int(1000 * position["bottom"] / page_size["height"]),
    ]


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
    for p in preds:
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
    empty_chunk_ratio=2.0,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,  # Previously 4
    num_train_epochs=16,
    fp16=True,
    model_name_or_path="microsoft/layoutlmv3-base",
    config_name=None,
    tokenizer_name=None,
    task_name="ner",
    cache_dir=None,
    model_revision="main",
    input_size=224,
    use_auth_token=False,
    visual_embed=True,
    imagenet_default_mean_and_std=False,
    train_interpolation="bicubic",
    output_dir=None,
    pad_to_max_length=True,
    overwrite_cache=True,
    preprocessing_num_workers=None,
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
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        config_name if config_name else model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task_name,
        cache_dir=cache_dir,
        revision=model_revision,
        input_size=input_size,
        use_auth_token=use_auth_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        tokenizer_file=None,  # avoid loading from a cached file of the pre-trained model in another machine
        cache_dir=cache_dir,
        use_fast=True,
        add_prefix_space=True,
        revision=model_revision,
        use_auth_token=use_auth_token,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir,
        revision=model_revision,
        use_auth_token=use_auth_token,
    )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if pad_to_max_length else False

    if visual_embed:
        mean = (
            IMAGENET_INCEPTION_MEAN
            if not imagenet_default_mean_and_std
            else IMAGENET_DEFAULT_MEAN
        )
        std = (
            IMAGENET_INCEPTION_STD
            if not imagenet_default_mean_and_std
            else IMAGENET_DEFAULT_STD
        )
        common_transform = Compose(
            [
                # transforms.ColorJitter(0.4, 0.4, 0.4),
                # transforms.RandomHorizontalFlip(p=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=input_size, interpolation=train_interpolation
                )
            ]
        )

        patch_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples, augmentation=False):
        tokenized_inputs = tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        labels = []
        bboxes = []
        images = []
        doc_paths = []
        token_offsets = []
        orig_labels = []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            offsets = tokenized_inputs.offset_mapping[batch_index]
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][
                batch_index
            ]
            label_ids = []
            bbox_inputs = []
            orig_labels_i = []
            doc_paths.append(examples["doc_path"][org_batch_index])
            chunk_doc_offsets = [
                {
                    "start": offset[0] + examples["page_offset"][org_batch_index],
                    "end": offset[1] + examples["page_offset"][org_batch_index],
                }
                for offset in offsets
            ]
            token_offsets.append(chunk_doc_offsets)
            assert len(word_ids) == len(chunk_doc_offsets)
            for i, (word_idx, token_doc_offset) in enumerate(
                zip(word_ids, chunk_doc_offsets)
            ):
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                else:
                    label_dicts = [
                        l
                        for l in examples["labels"][org_batch_index]
                        if overlaps(l, token_doc_offset)
                    ]
                    for l in label_dicts:
                        # Handles overlap
                        l["matched"] = True
                    if label_dicts:
                        orig_labels_i.append(label_dicts[0])
                        label = label_dicts[0]["label"]
                    else:
                        label = "<NONE>"
                    label_ids.append(label_to_id[label])

                    tol = 0
                    positions = []
                    while not positions:
                        if tol > 2:  # Tol of 2 is used commonly for whitespace tokens
                            print(
                                "Token did not match a position - using the previous token instead. Tol = {}".format(
                                    tol
                                )
                            )
                        positions = [
                            t["line_position"]
                            for t in examples["page_tokens"][org_batch_index]
                            if overlaps(
                                t["doc_offset"],
                                {
                                    "start": token_doc_offset["start"] - tol,
                                    "end": token_doc_offset["end"],
                                },
                            )
                        ]
                        tol += 1
                    if len(positions) > 1:
                        print("Warning - matched multiple position tokens")
                    pos = positions[-1]
                    bbox_inputs.append(
                        get_layoutlm_position(
                            pos, examples["page_size"][org_batch_index]
                        )
                    )
            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            orig_labels.append(orig_labels_i)

            if visual_embed:
                ipath = examples["page_image"][org_batch_index]
                img = pil_loader(ipath)
                for_patches, _ = common_transform(img, augmentation=augmentation)
                patch = patch_transform(for_patches)
                images.append(patch)
        for l_batch in examples["labels"]:
            for li in l_batch:
                assert li.get("matched", False)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["orig_labels"] = orig_labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["doc_path"] = doc_paths
        tokenized_inputs["token_offsets"] = token_offsets
        if visual_embed:
            tokenized_inputs["images"] = images

        return tokenized_inputs

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

    train_dataset = get_hf_dataset("train", is_training=True)
    eval_dataset = get_hf_dataset("val")

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if fp16 else None,
        padding=padding,
        max_length=512,
    )

    # Metrics
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
            return final_results

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **training_args,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
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
        print(predictions)
        pred_by_path = defaultdict(list)
        for prediction, example in zip(predictions, hf_dataset):
            preds = []
            for pred, offsets in zip(prediction, example["token_offsets"]):
                pred = label_list[pred]
                if pred != "<NONE>":
                    print("Adding non-none pred")
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
    del config["base_model"]
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
    project, entity, dataset_name, dataset_dir="/datasets",
):
    sweep_id_config = {
        "dataset_name": {"value": dataset_name},
        "dataset_dir": {"value": dataset_dir},
        "base_model": {"value": "layoutlmv3"},
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
                "max": 2,
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
