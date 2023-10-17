import wandb
import random
import os
import pandas as pd
import json
import tempfile
import traceback
import metrics
import tensorflow as tf


def get_num_runs(sweep_id, entity, project):
    sweep = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}")
    if sweep.state not in {"RUNNING", "PENDING"}:
        print("This sweep is currently {}".format(sweep.state))
        exit(0)
    num_runs = len([None for r in sweep.runs if r.state in {"finished", "running"}])
    print(f"Current number of runs {num_runs}")
    return num_runs


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


def get_sweep_config(training_lib):
    config = dict(**wandb.config)
    tl = config.pop("training_lib")
    assert tl == training_lib, f"You are trying to resume a {tl} using {training_lib}"
    return config


def run_agent(sweep_id, entity, project, train_and_predict, training_lib):
    def train_model():
        try:
            wandb.init(save_code=True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                train_and_predict(output_dir=tmp_dir, **get_sweep_config(training_lib))
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


def overlaps(a, b):
    return a["start"] < b["end"] <= a["end"] or b["start"] < a["end"] <= b["end"]


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
            if (
                page_do["start"] == page_do["end"]
                and page_do["start"]
                != doc_ocr[i - 1]["pages"][0]["doc_offset"]["end"] + 1
            ):
                consecutive += 1
                page_do["start"] += consecutive
                page_do["end"] += consecutive
            else:
                consecutive = 0
    return doc_ocr


def get_dataset_path(dataset_name, dataset_dir, split):
    return os.path.join(dataset_dir, dataset_name, f"{split}.csv")


def get_dataset(dataset_name, dataset_dir, split, is_document=None):
    if is_document is not None:
        csv_path = get_dataset_path(dataset_name, dataset_dir, split)
    else:
        csv_path = os.path.join(dataset_dir, dataset_name, f"{split}.csv")

    csv = pd.read_csv(csv_path)

    if is_document is not None:
        labels = [json.loads(l) for l in csv.labels]
        for t, l in zip(csv.text, labels):
            for li in l:
                assert t[li["start"] : li["end"]] == li["text"]
        return (
            get_model_input_from_csv(
                csv, is_document=is_document, dataset_dir=dataset_dir
            ),
            labels,
        )

    data = []
    for row in csv.to_dict("records"):
        item = {
            "text": row["text"],
            "labels": json.loads(row["labels"]),
            "doc_path": row["document_path"],
        }
        data.append(item)
    return data
