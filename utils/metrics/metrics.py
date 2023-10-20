import json
from collections import OrderedDict, defaultdict
from functools import partial

import fire
import numpy as np
import pandas as pd


def _get_unique_classes(true, predicted):
    true_and_pred = list(true) + list(predicted)
    return list(set([seq["label"] for seqs in true_and_pred for seq in seqs]))


def calc_recall(TP, FN):
    try:
        return TP / float(FN + TP)
    except ZeroDivisionError:
        return 0.0


def calc_precision(TP, FP):
    try:
        return TP / float(FP + TP)
    except ZeroDivisionError:
        return 0.0


def calc_f1(recall, precision):
    try:
        return 2 * (recall * precision) / (recall + precision)
    except ZeroDivisionError:
        return 0.0


def seq_recall(true, predicted, span_type):
    count_fn = get_seq_count_fn(span_type)
    class_counts = count_fn(true, predicted)
    results = {}
    for cls_, counts in class_counts.items():
        FN = len(counts["false_negatives"])
        TP = len(counts["true_positives"])
        results[cls_] = calc_recall(TP, FN)
    return results


def seq_precision(true, predicted, span_type):
    count_fn = get_seq_count_fn(span_type)
    class_counts = count_fn(true, predicted)
    results = {}
    for cls_, counts in class_counts.items():
        FP = len(counts["false_positives"])
        TP = len(counts["true_positives"])
        results[cls_] = calc_precision(TP, FP)
    return results


def seq_f1(true, predicted, span_type):
    count_fn = get_seq_count_fn(span_type)
    class_counts = count_fn(true, predicted)
    results = OrderedDict()
    for cls_, counts in class_counts.items():
        FP = len(counts["false_positives"])
        FN = len(counts["false_negatives"])
        TP = len(counts["true_positives"])
        recall = calc_recall(TP, FN)
        precision = calc_precision(TP, FP)
        results[cls_] = calc_f1(recall, precision)
    return results


def strip_whitespace(y):
    label_text = y["text"]
    lstripped = label_text.lstrip()
    new_start = y["start"] + (len(label_text) - len(lstripped))
    stripped = label_text.strip()
    return {
        "text": label_text.strip(),
        "start": new_start,
        "end": new_start + len(stripped),
        "label": y["label"],
    }


def sequences_overlap(x: dict, y: dict) -> bool:
    return x["start"] < y["end"] and y["start"] < x["end"]


def sequence_exact_match(true_seq, pred_seq):
    """
    Boolean return value indicates whether or not seqs are exact match
    """
    true_seq = strip_whitespace(true_seq)
    pred_seq = strip_whitespace(pred_seq)
    return pred_seq["start"] == true_seq["start"] and pred_seq["end"] == true_seq["end"]


def sequence_labeling_counts(true, predicted, equality_fn):
    """
    Return FP, FN, and TP counts
    """
    unique_classes = _get_unique_classes(true, predicted)

    d = {
        cls_: {"false_positives": [], "false_negatives": [], "true_positives": []}
        for cls_ in unique_classes
    }

    for i, (true_annotations, predicted_annotations) in enumerate(zip(true, predicted)):
        # add doc idx to make verification easier
        for annotations in [true_annotations, predicted_annotations]:
            for annotation in annotations:
                annotation["doc_idx"] = i

        for true_annotation in true_annotations:
            for pred_annotation in predicted_annotations:
                if equality_fn(true_annotation, pred_annotation):
                    if pred_annotation["label"] == true_annotation["label"]:
                        d[true_annotation["label"]]["true_positives"].append(
                            true_annotation
                        )
                        break
            else:
                d[true_annotation["label"]]["false_negatives"].append(true_annotation)

        for pred_annotation in predicted_annotations:
            for true_annotation in true_annotations:
                if (
                    equality_fn(true_annotation, pred_annotation)
                    and true_annotation["label"] == pred_annotation["label"]
                ):
                    break
            else:
                d[pred_annotation["label"]]["false_positives"].append(pred_annotation)

    return d


def get_seq_count_fn(span_type):
    span_type_fn_mapping = {
        "overlap": partial(sequence_labeling_counts, equality_fn=sequences_overlap),
        "exact": partial(sequence_labeling_counts, equality_fn=sequence_exact_match),
    }
    return span_type_fn_mapping[span_type]


def annotation_report(y_true, y_pred, labels=None, target_names=None, digits=2):
    # Adaptation of https://github.com/scikit-learn/scikit-learn/blob/f0ab589f/sklearn/metrics/classification.py#L1363
    exact_precision = seq_precision(y_true, y_pred, span_type="exact")
    exact_recall = seq_recall(y_true, y_pred, span_type="exact")
    exact_f1 = seq_f1(y_true, y_pred, span_type="exact")
    overlap_precision = seq_precision(y_true, y_pred, span_type="overlap")
    overlap_recall = seq_recall(y_true, y_pred, span_type="overlap")
    overlap_f1 = seq_f1(y_true, y_pred, span_type="overlap")

    count_dict = defaultdict(int)
    for annotation_seq in y_true:
        for annotation in annotation_seq:
            count_dict[annotation["label"]] += 1

    seqs = [
        exact_precision,
        exact_recall,
        exact_f1,
        overlap_precision,
        overlap_recall,
        overlap_f1,
        dict(count_dict),
    ]
    labels = set(exact_precision.keys()) | set(exact_recall.keys())
    target_names = ["%s" % l for l in labels]
    counts = [count_dict.get(target_name, 0) for target_name in target_names]

    last_line_heading = "Weighted Summary"
    headers = [
        "exact_precision",
        "exact_recall",
        "exact_f1",
        "overlap_precision",
        "overlap_recall",
        "overlap_f1",
        "support",
    ]
    label_width = max(len(tn) for tn in target_names + [last_line_heading])
    width = max(len(h) for h in headers)
    head_fmt = "{:>{label_width}s}" + " {:>{width}}" * len(headers)
    report = head_fmt.format("", *headers, label_width=label_width, width=width)
    report += "\n\n"
    row_fmt = "{:>{label_width}s}" + " {:>{width}.{digits}f}" * 6 + " {:>{width}}" "\n"
    seqs = [[seq.get(target_name, 0.0) for target_name in target_names] for seq in seqs]
    rows = zip(target_names, *seqs)
    for row in rows:
        report += row_fmt.format(
            *row, label_width=label_width, width=width, digits=digits
        )

    report += "\n"
    averages = [np.average(seq, weights=counts) for seq in seqs[:-1]] + [
        np.sum(seqs[-1])
    ]
    report += row_fmt.format(
        last_line_heading,
        *averages,
        label_width=label_width,
        width=width,
        digits=digits,
    )
    return report


def main(csv_path):
    df = pd.read_csv(csv_path)
    labels = [json.loads(l) for l in df.labels]
    preds = [json.loads(p) for p in df.preds]
    metrics = annotation_report(labels, preds)
    print(metrics)
    with open(csv_path + "_metrics.txt", "wt") as fp:
        fp.write(metrics)


def get_metrics_dict(csv_path, split):
    df = pd.read_csv(csv_path)
    labels = [json.loads(l) for l in df.labels]
    preds = [json.loads(p) for p in df.preds]
    metrics = dict()
    for span_type in ["overlap", "exact"]:
        for metric, metric_name in [
            (seq_f1, "f1"),
            (seq_precision, "precision"),
            (seq_recall, "recall"),
        ]:
            values = metric(labels, preds, span_type=span_type)
            for cls, value in values.items():
                metrics[
                    f"{split}_{span_type}_{cls}_{metric_name}".format(span_type, cls)
                ] = value
            metrics[f"{split}_macro_{metric_name}"] = float(
                np.average(list(values.values()))
            )
    return metrics


def cli():
    fire.Fire(main)


if __name__ == "__main__":
    cli()
