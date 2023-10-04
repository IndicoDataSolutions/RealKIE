import json
import gzip
import os
import pandas as pd
import random
from helpers import overlaps


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
