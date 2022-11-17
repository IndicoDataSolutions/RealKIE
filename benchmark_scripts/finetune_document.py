import json
import os

import fire
import pandas as pd

from finetune import DocumentLabeler
from finetune.base_models import LayoutLM

def get_dataset_path(dataset_name, split, order_by):
    return f"{dataset_name}/{split}_w_sparse_tables_ordered_by_{order_by}.csv"

def get_dataset(dataset_name, split, order_by):
    csv = pd.read_csv(get_dataset_path(dataset_name, split, order_by))
    return [json.loads(o) for o in csv.ocr], [json.loads(l) for l in csv.labels]

def train_and_eval_model(
        dataset, class_weights="sqrt", collapse_whitespace=True, optimize_for="accuracy", order_by="cols"
):
    model = DocumentLabeler(
        base_model=LayoutLM,
        class_weights=class_weights,
        collapse_whitespace=collapse_whitespace,
        optimize_for=optimize_for,
    )
    model.fit(*get_dataset(dataset, "train", order_by))
    test_data = pd.read_csv(get_dataset_path(dataset, "test", order_by=order_by))
    preds = model.predict([json.loads(o) for o in test_data.ocr])
    test_data["preds"] = [json.dumps(p) for p in preds]
    results_dir = os.path.join("finetune", dataset)
    os.makedirs(results_dir, exist_ok=True)
    test_data.to_csv(os.path.join(results_dir, f"predictions_document_{order_by}.csv"))

if __name__ == "__main__":
    fire.Fire(train_and_eval_model)
