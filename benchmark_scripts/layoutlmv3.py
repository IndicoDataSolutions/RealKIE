#!/usr/bin/env python
# coding=utf-8
# Based on https://github.com/microsoft/unilm/blob/master/layoutlmv3/examples/run_funsd_cord.py
import logging
import os
import fire
import pandas as pd
import json
from collections import defaultdict

import numpy as np
from datasets import load_metric, Dataset

import transformers

from layoutlmft.data import DataCollatorForKeyValueExtraction
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)

from layoutlmft.data.image_utils import (
    RandomResizedCropAndInterpolationWithTwoPic,
    pil_loader,
    Compose,
)

from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from torchvision import transforms
import torch


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
        "top":	min(p["top"] for p in positions),
        "bottom": max(p["bottom"] for p in positions),
    }

def add_line_bboxes(page_text, page_tokens):
    lines = page_text.split("\n")
    line_spans = []
    for line in lines:
        start = line_spans[-1]["end"] + 1 if line_spans else 0
        line_spans.append(
            {
                "start": start,
                "end": start + len(line)
            }   
        )
        line_tokens = [tok for tok in page_tokens if overlaps(tok["page_offset"], line_spans[-1])]
        line_spans[-1]["position"] = join_positions(
            [tok["position"] for tok in line_tokens]
        )
        for tok in line_tokens:
            tok["line_position"] = line_spans[-1]["position"]
    
def get_dataset(dataset_name, split):
    csv = pd.read_csv(f"{dataset_name}/{split}.csv")
    data = []
    for row in csv.to_dict("records"):
        labels = json.loads(row["labels"])
        ocr = json.loads(row["ocr"])
        images = json.loads(row["image_files"])
        for page_ocr, page_image in zip(ocr, images):
            doc_offset = page_ocr["pages"][0]["doc_offset"]
            page_labels = [l for l in labels if overlaps(l, doc_offset)]
            add_line_bboxes(page_ocr["pages"][0]["text"], page_ocr["tokens"])
            item = {
                "text": page_ocr["pages"][0]["text"],
                "labels": page_labels,
                "page_tokens": page_ocr["tokens"],
                "page_image": page_image,
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

def clean_preds(preds, char_threshold=1):
    preds = sorted(preds, key=lambda x: x["start"])
    output = []
    for p in preds:
        if output and p["start"] - output[-1]["end"] <= char_threshold and p["label"] == output[-1]["label"]:
            output["end"] = p["end"]
        else:
            output.append(p)
    return output

def main(
    dataset_name,
    per_device_train_batch_size=2,
    num_train_epochs=8,
    n_gpu=1,
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
    **training_args
):
    if output_dir is None:
        output_dir = os.path.join("layoutlmv3", dataset_name, "model_output")
    train_dataset = get_dataset(dataset_name, "train")
    label_list = ["<NONE>"] + sorted(
        set(l["label"] for page in train_dataset for l in page["labels"] if l["label"] != "<NONE>")
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

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
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
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
#            is_split_into_words=True,
            # TODO: add stride in here??
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
            token_offsets.append(
                [
                    {
                        "start": offset[0] + examples["page_offset"][org_batch_index],
                        "end": offset[1] + examples["page_offset"][org_batch_index]
                    } for offset in offsets
                ]
            )
            assert len(word_ids) == len(offsets)
            for i, (word_idx, offset) in enumerate(zip(word_ids, offsets)):
                offset = {"start": offset[0], "end": offset[1]}
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                assert offset["end"] <= len(examples["text"][org_batch_index]), (offset, len(examples["text"][org_batch_index]))
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                else:
                    label_dicts = [
                        l for l in examples["labels"][org_batch_index] if overlaps(l, offset)
                    ]
                    if label_dicts:
                        label = label_dicts[0]["label"]
                        print("Found match for label {}".format(label))
                    else:
                        label = "<NONE>"
                    label_ids.append(label_to_id[label])

                    tol = 0
                    positions = []
                    while not positions:
                        if tol > 1:
                            print("Token did not match a position - using the previous token instead. Tol = {}".format(tol))
                        positions = [
                            t["line_position"]
                            for t in examples["page_tokens"][org_batch_index]
                            if overlaps(t["page_offset"], {"start": offset["start"] - tol, "end": offset["end"]})
	                ]
                        tol += 1
                    if len(positions) > 1:
                        print("Warning - matched multiple position tokens")
                    pos = positions[-1]
                    bbox_inputs.append(
                        get_layoutlm_position(pos, examples["page_size"][org_batch_index])
                    )  # TODO: this should actually be line bboxes normalized as ints 1-1000.
            labels.append(label_ids)
            bboxes.append(bbox_inputs)

            if visual_embed:
                ipath = examples["page_image"][org_batch_index]
                img = pil_loader(ipath)
                for_patches, _ = common_transform(img, augmentation=augmentation)
                patch = patch_transform(for_patches)
                images.append(patch)
        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["doc_path"] = doc_paths
        tokenized_inputs["token_offsets"] = token_offsets
        if visual_embed:
            tokenized_inputs["images"] = images

        return tokenized_inputs

    def get_hf_dataset(split):
        split_dataset = Dataset.from_generator(
            lambda: get_dataset(dataset_name, "train")
        )
        return  split_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=split_dataset.column_names, # Drop all existing columns so that we can change the lengths of the output
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
        )
    train_dataset = get_hf_dataset("train")
    test_dataset = get_hf_dataset("test")
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
            **training_args
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,)
        

    # Training
    train_result = trainer.train(resume_from_checkpoint=None)
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

    logger.info("*** Predict ***")

    predictions, labels, metrics = trainer.predict(test_dataset)
    predictions = np.argmax(predictions, axis=2)

    pred_by_path = defaultdict(list)
    for prediction, example in zip(predictions, test_dataset):
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
        preds_by_path[example["doc_path"]] += preds
    pred_records = []
    for doc_path, preds in pred_by_path.items():
        pred_records.append(
            {
                "doc_path": doc_path,
                "preds": clean_preds(preds),
            }
        )
    pd.DataFrame.from_records(pred_records).to_csv(os.path.join(output_dir, "predictions.csv"))

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    # Save predictions
    output_test_predictions_file = os.path.join(output_dir, "test_predictions.txt")
    with open(output_test_predictions_file, "w") as writer:
        for prediction in true_predictions:
            writer.write(" ".join(prediction) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
