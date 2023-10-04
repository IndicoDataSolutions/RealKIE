# Based on https://github.com/microsoft/unilm/blob/master/layoutlmv3/examples/run_funsd_cord.py
import functools
import json
import logging
import os
import tempfile
import traceback
from collections import defaultdict

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
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.utils import check_min_version

from datasets import Dataset, load_metric, disable_caching

from helpers import (
    get_matching_sweep,
    get_num_runs,
    clean_preds,
    undersample_empty_chunks,
    get_dataset,
    overlaps,
)
from layoutlm_helpers import (
    get_dataset_as_pages,
    get_layoutlm_position,
)
from all_sweep_configs import get_configs_for_model

disable_caching()
from metrics import metrics

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)

TRAINING_LIB = "layoutlmv3"


def train_and_predict(
    dataset_name,
    dataset_dir="/datasets",
    empty_chunk_ratio=2.0,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=16,
    fp16=True,
    model_name_or_path="microsoft/layoutlmv3-base",
    input_size=224,
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
    label_list = ["<NONE>"] + sorted(
        set(
            l["label"]
            for page in get_dataset(dataset_name, "train", dataset_dir)
            for l in page["labels"]
            if l["label"] != "<NONE>"
        )
    )
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id,
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
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            offsets = tokenized_inputs.offset_mapping[batch_index]
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][
                batch_index
            ]
            label_ids = []
            bbox_inputs = []
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
                        label = label_dicts[0]["label"]
                    else:
                        label = "<NONE>"
                    label_ids.append(label_to_id[label])
                    tol = 0
                    positions = []
                    while not positions:
                        t_start = max(0, token_doc_offset["start"] - tol)
                        t_end = token_doc_offset["end"] + tol
                        do = examples["page_offset"][org_batch_index]
                        t_text = examples["text"][org_batch_index][
                            t_start - do : t_end - do
                        ]
                        examples["text"]
                        if (
                            tol > 2
                            and len(t_text.replace(" ", "").replace("\n", "")) > 3
                        ):  # Tol of 2 is used commonly for whitespace tokens
                            print(
                                "Token did not match a position - using the previous token instead. Tol = {}".format(
                                    tol
                                )
                            )
                        positions = [
                            t["line_position"]
                            for t in examples["page_tokens"][org_batch_index]
                            if overlaps(
                                t["doc_offset"], {"start": t_start, "end": t_end}
                            )
                        ]
                        tol += 1
                    pos = positions[-1]
                    bbox_inputs.append(
                        get_layoutlm_position(
                            pos, examples["page_size"][org_batch_index]
                        )
                    )
            labels.append(label_ids)
            bboxes.append(bbox_inputs)

            if visual_embed:
                ipath = examples["page_image"][org_batch_index]
                img = pil_loader(ipath)
                for_patches, _ = common_transform(img, augmentation=augmentation)
                patch = patch_transform(for_patches)
                images.append(patch)
        unmatched = []
        for l_batch in examples["labels"]:
            for li in l_batch:
                if not li.get("matched", False):
                    unmatched.append(li)
        assert len(unmatched) == 0, unmatched

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["doc_path"] = doc_paths
        tokenized_inputs["token_offsets"] = token_offsets
        if visual_embed:
            tokenized_inputs["images"] = images

        return tokenized_inputs

    def get_hf_dataset(split, is_training=False):
        split_dataset = Dataset.from_generator(
            lambda: get_dataset_as_pages(dataset_name, split, dataset_dir=dataset_dir)
        )
        tokenized = split_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=split_dataset.column_names,  # Drop all existing columns so that we can change the lengths of the output
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
        )
        if is_training and empty_chunk_ratio is not None:
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


def get_configs_for_model():
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
                train_and_predict(output_dir=tmp_dir, **get_configs_for_model())
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


def setup_and_run_sweep(project, entity, dataset_name, dataset_dir="/datasets"):
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
    sweep_configs = get_configs_for_model("layoutlm", sweep_id_config)
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
