import os
import gzip
import json

from helpers import fix_page_offsets, strip_unused_ocr_data

from finetune.base_models import (
    BERTLarge,
    BERTModelCased,
    LayoutLM,
    RoBERTa,
    ROBERTALarge,
    XDocBase,
)
from finetune import DocumentLabeler, SequenceLabeler


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


def get_model_input_from_csv(csv, is_document, dataset_dir):
    if is_document:
        ocr = []
        for ocr_file, text in zip(csv.ocr, csv.text):
            ocr_file = os.path.join(dataset_dir, ocr_file)
            with gzip.open(ocr_file, "rt") as fp:
                doc_ocr = json.loads(fp.read())
                doc_ocr = fix_page_offsets(
                    [strip_unused_ocr_data(ocr_page) for ocr_page in doc_ocr]
                )
                ocr.append(doc_ocr)
            assert "\n".join(page["pages"][0]["text"] for page in ocr[-1]) == text
        return ocr
    return csv.text
