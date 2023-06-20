import json
import os
import tempfile
import traceback
import gzip

import fire
import pandas as pd
import wandb

import tensorflow as tf

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


def strip_unused_ocr_data(ocr_data: dict):
    ocr_data.pop("chars", None)
    ocr_data.pop("blocks", None)
    for token in ocr_data["tokens"]:
        token.pop("style", None)
        token.pop("page_offset", None)
        token.pop("block_offset", None)
        token.pop("page_num", None)
    for page in ocr_data["pages"]:
        page.pop("image", None)
        page.pop("thumbnail", None)
        page.pop("ocr_statistics", None)
        page.pop("page_num", None)
    return ocr_data

def fix_page_offsets(doc_ocr):
    # This doesn't actually change the data in any important way,
    # but just stops us hitting assertion errors in finetune
    # Error comes from long strings of empty pages in resource contracts.
    consecutive = 0
    for i, page in enumerate(doc_ocr):
        if i >= 0:
            page_do = page["pages"][0]["doc_offset"]
            if page_do["start"] == page_do["end"]:
                consecutive += 1
                page_do["start"] += consecutive
                page_do["end"] += consecutive
            else:
                consecutive = 0
    return doc_ocr

def get_model_input_from_csv(csv, is_document, dataset_dir):
    if is_document:
        ocr = []
        for ocr_file, text in zip(csv.ocr, csv.text):
            ocr_file = os.path.join(dataset_dir, ocr_file)
            with gzip.open(ocr_file, 'rt') as fp:
                doc_ocr = json.loads(fp.read())
                doc_ocr = fix_page_offsets([strip_unused_ocr_data(ocr_page) for ocr_page in doc_ocr])
                ocr.append(doc_ocr)
            assert "\n".join(page["pages"][0]["text"] for page in ocr[-1]) == text
        return ocr
    return csv.text


def get_dataset(dataset_name, dataset_dir, split, is_document):
    csv = pd.read_csv(get_dataset_path(dataset_name, dataset_dir, split))
    labels = [json.loads(l) for l in csv.labels]
    for t, l in zip(csv.text, labels):
        for li in l:
            assert t[li["start"]: li["end"]] == li["text"]
            #del li["text"] # Label checking inside finetune assumes labels cannot span pages.
    return get_model_input_from_csv(csv, is_document=is_document, dataset_dir=dataset_dir), labels


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
    preds = model.predict(get_model_input_from_csv(test_data, is_document=is_document, dataset_dir=dataset_dir))
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
            "auto_negative_sampling": {"values": [False, True]},
            "max_empty_chunk_ratio": {
                "distribution": "log_uniform_values",
                "min": 1e-2,
                "max": 1000.0,
            },
            "lr": {"distribution": "log_uniform_values", "min": 1e-8, "max": 1e-2},
            "batch_size": {"distribution": "int_uniform", "min": 1, "max": 8},
            "n_epochs": {"distribution": "int_uniform", "min": 1, "max": 16},
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
    return run_agent(sweep_id, entity=entity, project=project)


if __name__ == "__main__":
    fire.Fire(
        {
            "train_and_predict": train_and_predict,
            "wandb_sweep": setup_and_run_sweep,
            "wandb_resume_sweep": run_agent,
        }
    )
