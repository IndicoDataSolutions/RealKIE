import json
import gzip
import os
import pandas as pd
import numpy as np
from collections import defaultdict

from helpers import overlaps, get_dataset, clean_preds
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from datasets import Dataset, load_metric, disable_caching

disable_caching()


import functools

import torch

from torchvision import transforms
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


class LayoutLMLoader:
    def __init__(
        self,
        dataset_name,
        dataset_dir,
        base_model,
        preprocessing_num_workers,
        overwrite_cache,
        empty_chunk_ratio,
        imagenet_default_mean_and_std,
        input_size,
        visual_embed,
        train_interpolation,
    ):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.base_model = base_model
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        self.empty_chunk_ratio = empty_chunk_ratio
        self.imagenet_default_mean_and_std = imagenet_default_mean_and_std
        self.input_size = (input_size,)
        self.visual_embed = (visual_embed,)
        self.train_interpolation = train_interpolation

        self.train_dataset = self.get_dataset("train")
        self.label_list = self.get_label_list()
        self.label_to_id = {l: i for i, l in enumerate(self.label_list)}
        self.id_to_label = {i: l for i, l in enumerate(self.label_list)}

        self.model, self.tokenizer = self.initialize_model_and_tokenizer()

    def initialize_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(
            self.base_model,
            num_labels=self.num_labels,
            id2label=self.id_to_label,
            label2id=self.label_to_id,
        )
        return model, tokenizer

    def tokenize_and_align_labels(self, examples, visual_embed, augmentation=False):
        tokenized_inputs = self.tokenizer(
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
                    label_ids.append(self.label_to_id[label])
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

            if self.visual_embed:
                mean = (
                    IMAGENET_INCEPTION_MEAN
                    if not self.imagenet_default_mean_and_std
                    else IMAGENET_DEFAULT_MEAN
                )
                std = (
                    IMAGENET_INCEPTION_STD
                    if not self.imagenet_default_mean_and_std
                    else IMAGENET_DEFAULT_STD
                )
                common_transform = Compose(
                    [
                        # transforms.ColorJitter(0.4, 0.4, 0.4),
                        # transforms.RandomHorizontalFlip(p=0.5),
                        RandomResizedCropAndInterpolationWithTwoPic(
                            size=self.input_size, interpolation=self.train_interpolation
                        )
                    ]
                )

                patch_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=torch.tensor(mean), std=torch.tensor(std)
                        ),
                    ]
                )
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

    def get_hf_dataset(self, split, is_training=False):
        split_dataset = Dataset.from_generator(
            lambda: get_dataset_as_pages(self.dataset_name, split)
        )
        tokenized = split_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=split_dataset.column_names,  # Drop all existing columns so that we can change the lengths of the output
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=not self.overwrite_cache,
        )
        if is_training and self.empty_chunk_ratio is not None:
            tokenized = tokenized.map(
                functools.partial(
                    self.undersample_empty_chunks,
                    empty_chunk_ratio=self.empty_chunk_ratio,
                ),
                batched=True,
                remove_columns=tokenized.column_names,
                num_proc=self.preprocessing_num_workers,
                load_from_cache_file=not self.overwrite_cache,
                batch_size=1000,
            )
        return tokenized

    def compute_metrics(self, p):
        metric = load_metric("seqeval")

        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
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

    def run_predictions(self, trainer, split, output_path):
        hf_dataset = self.get_hf_dataset(split)
        orig_dataset = get_dataset(self.dataset_name, split, self.dataset_dir)

        predictions, _, _ = trainer.predict(hf_dataset)
        predictions = np.argmax(predictions, axis=2)
        pred_by_path = defaultdict(list)
        for prediction, example in zip(predictions, hf_dataset):
            preds = []
            for pred, offsets in zip(prediction, example["token_offsets"]):
                pred = self.label_list[pred]
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


def get_dataset_as_pages(dataset_name, split, dataset_dir):
    csv = pd.read_csv(os.path.join(dataset_dir, dataset_name, f"{split}.csv"))
    data = []
    for i, row in enumerate(csv.to_dict("records")):
        labels = json.loads(row["labels"])
        with gzip.open(os.path.join(dataset_dir, row["ocr"]), "rt") as fp:
            ocr = json.loads(fp.read())
        images = json.loads(row["image_files"])
        for page_ocr, page_image in zip(ocr, images):
            doc_offset = page_ocr["pages"][0]["doc_offset"]
            # Skip labels for empty pages.
            page_labels = [
                l
                for l in labels
                if overlaps(l, doc_offset)
                and len(page_ocr["pages"][0]["text"].strip()) > 0
            ]
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


def clip_position(pos):
    return min(max(int(pos), 0), 1000)


def get_layoutlm_position(position, page_size):
    return [
        clip_position(1000 * position["left"] / page_size["width"]),
        clip_position(1000 * position["top"] / page_size["height"]),
        clip_position(1000 * position["right"] / page_size["width"]),
        clip_position(1000 * position["bottom"] / page_size["height"]),
    ]
