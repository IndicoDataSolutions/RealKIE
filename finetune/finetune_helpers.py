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
