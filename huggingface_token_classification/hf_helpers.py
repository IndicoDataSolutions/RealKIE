from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from datasets import Dataset

import evaluate
from helpers import overlaps, get_dataset, undersample_empty_chunks, clean_preds
import numpy as np
import functools
from collections import defaultdict
import json
import pandas as pd


class HFLoader:
    def __init__(
        self,
        dataset_name,
        dataset_dir,
        base_model,
        preprocessing_num_workers,
        overwrite_cache,
        empty_chunk_ratio,
    ):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.base_model = base_model
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        self.empty_chunk_ratio = empty_chunk_ratio

        self.train_dataset = self.get_dataset("train")
        self.label_list = self.get_label_list()
        self.label_to_id = {l: i for i, l in enumerate(self.label_list)}
        self.id_to_label = {i: l for i, l in enumerate(self.label_list)}

        self.model, self.tokenizer = self.initialize_model_and_tokenizer()

    def get_label_list(self, train_dataset):
        return ["<NONE>"] + sorted(
            set(
                l["label"]
                for page in train_dataset
                for l in page["labels"]
                if l["label"] != "<NONE>"
            )
        )

    def initialize_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if "deberta" in self.base_model:
            tokenizer.model_max_length = 512
        model = AutoModelForTokenClassification.from_pretrained(
            self.base_model,
            num_labels=len(self.label_list),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
        )
        return model, tokenizer

    def get_hf_dataset(
        self,
        split,
        dataset_name,
        dataset_dir,
        preprocessing_num_workers,
        overwrite_cache,
        empty_chunk_ratio,
        is_training=False,
    ):
        split_dataset = Dataset.from_generator(
            lambda: get_dataset(dataset_name, split, dataset_dir=dataset_dir)
        )
        tokenized = split_dataset.map(
            self.tokenize_and_align_labels,
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

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
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
                    label_ids.append(self.label_to_id[label])

            labels.append(label_ids)
        for doc_labels in examples["labels"]:
            for l in doc_labels:
                if not l.get("matched", False):
                    print(f"Unmatched Label {l}")
        tokenized_inputs["labels"] = labels
        tokenized_inputs["doc_path"] = doc_paths
        tokenized_inputs["token_offsets"] = token_offsets
        return tokenized_inputs

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        seqeval = evaluate.load("seqeval")
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

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
