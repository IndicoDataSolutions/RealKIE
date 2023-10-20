import os
import json
import gzip

import fire
import pandas as pd
from utils.helpers import overlaps

from finetune import SequenceLabeler


def get_qa_data(dataset_dir, dataset_name):
    df = pd.DataFrame()
    for split in ["train", "test", "val"]:
        split_df = pd.read_csv(os.path.join(dataset_dir, dataset_name, f"{split}.csv"))
        split_df["split"] = [split] * len(split_df)
        df = pd.concat([df, split_df])
    return df[["text", "labels", "document_path", "split", "ocr", "original_filename"]]


def get_ocr(ocr_file, dataset_dir):
    with gzip.open(os.path.join(dataset_dir, ocr_file), "rt") as fp:
        return json.loads(fp.read())


def main(dataset_name, dataset_dir="datasets"):
    df = get_qa_data(dataset_dir=dataset_dir, dataset_name=dataset_name)
    model_path = os.path.join(dataset_dir, dataset_name, f"qa_model.jl")
    if os.path.exists(model_path):
        model = SequenceLabeler.load(model_path)
    else:
        model = SequenceLabeler(auto_negative_sampling=False, class_weights="sqrt")
        model.fit(df.text, [json.loads(l) for l in df.labels])
        model.save(model_path)
    preds = model.predict(df.text)
    qa_df_records = []
    for p, row in zip(preds, df.to_dict("records")):
        page_boundaries = []
        for r in get_ocr(row["ocr"], dataset_dir):
            page_boundaries.append(
                {**r["pages"][0]["doc_offset"], "page_num": r["pages"][0]["page_num"]}
            )
        labels = json.loads(row["labels"])
        for pi in p:
            for li in labels:
                if pi["label"] == li["label"] and overlaps(pi, li):
                    pi["matched"] = True
                    li["matched"] = True
        for pi in p:
            if not pi.get("matched", False):
                qa_df_records.append(
                    {
                        "document_path": row["document_path"],
                        "filename": row["original_filename"],
                        "split": row["split"],
                        "error_type": "missed label",
                        "confidence": pi["confidence"][pi["label"]],
                        "text": pi["text"],
                        "text_w_context": row["text"][
                            pi["start"] - 20 : pi["end"] + 20
                        ],
                        "page_num": [pb for pb in page_boundaries if overlaps(pb, pi)][
                            0
                        ]["page_num"]
                        + 1,
                        "label": pi["label"],
                    }
                )
        for li in labels:
            if not li.get("matched", False):
                qa_df_records.append(
                    {
                        "document_path": row["document_path"],
                        "filename": row["original_filename"],
                        "split": row["split"],
                        "error_type": "extra label",
                        "confidence": "n/a",
                        "text": li["text"],
                        "text_w_context": row["text"][
                            li["start"] - 20 : li["end"] + 20
                        ],
                        "page_num": [pb for pb in page_boundaries if overlaps(pb, li)][
                            0
                        ]["page_num"]
                        + 1,
                        "label": li["label"],
                    }
                )
        pd.DataFrame.from_records(qa_df_records).to_csv(
            os.path.join(dataset_dir, dataset_name, "qa.csv")
        )


if __name__ == "__main__":
    fire.Fire(main)
