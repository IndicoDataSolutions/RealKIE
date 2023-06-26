import pandas as pd
import json
import fire
import os
import tqdm
import glob
from collections import defaultdict, Counter


def get_all_data(dataset_dir, dataset_name):
    df = pd.DataFrame()
    for split in ["train", "test", "val"]:
        split_df = pd.read_csv(os.path.join(dataset_dir, dataset_name, f"{split}.csv"))
        df = pd.concat([df, split_df])
    return df


def min_max_mean(values):
    return (min(values), max(values), sum(values) / len(values))


def list_labels(dataset_name, dataset_dir="datasets"):
    df = get_all_data(dataset_dir=dataset_dir, dataset_name=dataset_name)
    counts = defaultdict(int)
    for labels in df.labels:
        for l in json.loads(labels):
            counts[l["label"]] += 1
    for label, count in sorted(counts.items(), key=lambda v: v[1]):
        print(f"{label} = {count}")
    df_raw = pd.read_csv(os.path.join(dataset_dir, dataset_name, "raw_export.csv"))
    print(f"Number of docs in raw export {len(df_raw)}")
    print(f"Number of docs {len(df.labels)}")
    print(f"Number of unique docs {len(set(df.text))}")
    print(
        f"Min Max Mean Pages {min_max_mean([len(json.loads(images)) for images in df.image_files])}"
    )
    print(f"Min Max Mean Words {min_max_mean([len(text.split()) for text in df.text])}")


### Paper statistics


def entity_name(*, label_name, **kwargs):
    return label_name


def total_count(*, label_name, labels, **kwargs):
    return len([li for label in labels for li in label if li["label"] == label_name])


def unique_values(*, label_name, labels, **kwargs):
    return len(
        set(
            [
                li["text"].strip()
                for label in labels
                for li in label
                if li["label"] == label_name
            ]
        )
    )


def percent_docs_contained(*, label_name, labels, **kwargs):
    return sum(
        1 if any(l["label"] == label_name for l in label) else 0 for label in labels
    ) / len(labels)


def mean_entities_per_doc(*, label_name, labels, **kwargs):
    return sum(
        [len([l for l in label if l["label"] == label_name]) for label in labels]
    ) / len(labels)


def format_cell(cell):
    if isinstance(cell, float):
        return "{:.2f}".format(cell)
    return str(cell).replace("_", "\\_")


def make_latex_table(data, row_headers, identifier) -> str:
    header_row = " & ".join(row_headers) + " \\\\"
    rows = [header_row]

    for group_header, group_data in data.items():
        group_data = sorted(group_data, key=lambda x: x["Entity Name"])
        rows.append("\\midrule")
        rows.append(
            "\\multicolumn{5}{c}{\\emph{" + format_cell(group_header) + "}} \\\\"
        )
        rows.append("\\midrule")
        for entry in group_data:
            rows.append(
                " & ".join(format_cell(entry[h]) for h in row_headers) + " \\\\"
            )
    rows.append("\\bottomrule")

    latex_table = (
        "\\begin{tabular}{p{5,0cm}llll} \n" + "\n".join(rows) + "\n\\end{tabular}"
    )
    # latex_table = latex_table.replace('_', '\\_')
    latex_table = (
        latex_table
        + "\n}\n\\caption{Summary of something \\label{tab:"
        + identifier
        + "}}\n\\end{table*}"
    )
    latex_table = (
        "\\begin{table*}[htbp]\n\\setlength{\\tabcolsep}{2pt}\n\\centering\n\\scriptsize{"
        + latex_table
    )
    return latex_table


def label_statistics(dataset_dir="datasets"):
    columns = {
        "Entity Name": entity_name,
        "Total Count": total_count,
        "Unique Values": unique_values,
        "Docs Contained": percent_docs_contained,
        "Mean Entities / Doc": mean_entities_per_doc,
    }
    tables = []
    for group, identifier in [
        (["fcc_invoices", "s1", "resource_contracts"], "entities-1"),
        (["charities", "nda"], "entities-1"),
    ]:
        output = dict()
        for dataset_name in group:
            all_data = get_all_data(dataset_dir, dataset_name)
            dataset_values = list()
            labels = [json.loads(label) for label in all_data.labels]
            label_names = set(l["label"] for label in labels for l in label)
            for l in label_names:
                dataset_values.append(
                    {k: fn(labels=labels, label_name=l) for k, fn in columns.items()}
                )
            output[dataset_name] = dataset_values
        tables.append(
            make_latex_table(
                output, identifier=identifier, row_headers=list(columns.keys())
            )
        )
    print("\n".join(tables))


### End Paper Statistics

#### START FROM CHRIS #####
def summary_stats(by_type):
    count_dict = counts(by_type)
    lengths = text_lengths(by_type)
    summary_by_label = dict()
    for label, counts_per_doc in count_dict.items():
        minimum = min(counts_per_doc)
        maximum = max(counts_per_doc)
        mean = np.mean(counts_per_doc)
        summary_by_label[label] = {
            "min": minimum,
            "mean": mean,
            "max": maximum,
            "min_length": min(lengths[label])[0],
            "mean_length": np.mean(lengths[label]),
            "max_length": max(lengths[label])[0],
        }
    return pd.DataFrame(
        dict(sorted(summary_by_label.items(), key=lambda x: x[0].lower()))
    )


def text_lengths(by_type):
    text_lengths_dict = dict()
    for label, label_list in by_type.items():
        text_lengths_dict[label] = []
        for labels in label_list:
            for l in labels:
                spans = l["spans"]
                text_lengths_dict[label].append([s["end"] - s["start"] for s in spans])
    return text_lengths_dict


def plot_lengths():
    summary.T[["mean", "mean_length"]].plot.bar(subplots=True, log=True, legend=False)
    plt.show()


def plot_doc_length_histogram(data_set_id):
    client = create_client(HOST, API_TOKEN_PATH)
    d = indico_wrapper.Datasets(client)
    thing = d.get_dataset_metadata(data_set_id)
    lengths = [t["numPages"] for t in thing if t["name"][-4:].lower() == ".pdf"]
    plt.hist(lengths)
    plt.yscale("log")
    plt.show()


def summary_stats(by_type):
    count_dict = counts(by_type)
    lengths = text_lengths(by_type)
    summary_by_label = dict()
    for label, counts_per_doc in count_dict.items():
        minimum = min(counts_per_doc)
        maximum = max(counts_per_doc)
        mean = np.mean(counts_per_doc)
        summary_by_label[label] = {
            "min": minimum,
            "mean": mean,
            "max": maximum,
            "min_length": min(lengths[label])[0],
            "mean_length": np.mean(lengths[label]),
            "max_length": max(lengths[label])[0],
        }
    return pd.DataFrame(
        dict(sorted(summary_by_label.items(), key=lambda x: x[0].lower()))
    )


### END FROM CHRIS ###
def generate_plots(dataset_name, dataset_dir="datasets"):
    # TODO: write me

    # Table of label frequency, average length, min length, max length.
    # Table of doc lengths in chars, words, pages.
    pass


def drop_label_from_dataset(dataset_name, label_name, dataset_dir="datasets"):
    for split in ["train", "test", "val"]:
        print(f"Processing split {split}")
        split_path = os.path.join(dataset_dir, dataset_name, f"{split}.csv")
        split_df = pd.read_csv(split_path)
        cleaned_labels = []
        for row_labels in tqdm.tqdm(split_df.labels):
            cleaned_labels.append(
                json.dumps(
                    [l for l in json.loads(row_labels) if l["label"] != label_name]
                )
            )
        split_df["labels"] = cleaned_labels
        split_df.to_csv(split_path)


def rename_label(dataset_name, label_name, new_label_name, dataset_dir="datasets"):
    for split in ["train", "test", "val"]:
        print(f"Processing split {split}")
        split_path = os.path.join(dataset_dir, dataset_name, f"{split}.csv")
        split_df = pd.read_csv(split_path)
        cleaned_labels = []
        for row_labels in tqdm.tqdm(split_df.labels):
            cleaned_labels.append(
                json.dumps(
                    [
                        (
                            l
                            if l["label"] != label_name
                            else {**l, "label": new_label_name}
                        )
                        for l in json.loads(row_labels)
                    ]
                )
            )
        split_df["labels"] = cleaned_labels
        split_df.to_csv(split_path)


def cleanup_files(dataset_name, dataset_dir="datasets"):
    image_files = []
    pdf_files = []
    ocr_files = []
    for split in ["train", "test", "val"]:
        split_path = os.path.join(dataset_dir, dataset_name, f"{split}.csv")
        split_df = pd.read_csv(split_path)
        ocr_files += list(split_df.ocr)
        pdf_files += list(split_df.document_path)
        for row_images in split_df.image_files:
            image_files += json.loads(row_images)

    def cleanup(dir, expected):
        assert len(expected) == len(set(expected))  # No duplicate files referenced
        all_files = set(glob.glob(os.path.join(dir, "*")))
        for e in expected:
            e_full_path = os.path.join(dataset_dir, e)
            assert e_full_path in all_files, (e, all_files)
            all_files.remove(e_full_path)
        for f in all_files:
            os.remove(f)

    cleanup(os.path.join(dataset_dir, dataset_name, "ocr"), ocr_files)
    cleanup(os.path.join(dataset_dir, dataset_name, "files"), pdf_files)
    cleanup(os.path.join(dataset_dir, dataset_name, "images", "*"), image_files)
    for image_dir in glob.glob(os.path.join(dataset_dir, dataset_name, "images", "*")):
        # Drop any empty image folders.
        if len(os.listdir(image_dir)) == 0:
            os.rmdir(image_dir)


def deduplicate(dataset_name, dataset_dir="datasets"):
    all_texts = set()
    # Preference keeping test and val sets, drop dupes from train.
    for split in ["test", "val", "train"]:
        split_path = os.path.join(dataset_dir, dataset_name, f"{split}.csv")
        split_df = pd.read_csv(split_path)
        new_records = []
        for row in split_df.to_dict("records"):
            if row["text"] in all_texts:
                continue
            all_texts.add(row["text"])
            new_records.append(row)
        pd.DataFrame.from_records(new_records).to_csv(split_path)


def remove_leading_path_names(dataset_name, dataset_dir="datasets"):
    def strip_path(p):
        if f"{dataset_name}_v2/" in p:
            p = p.replace(f"{dataset_name}_v2/", f"{dataset_name}/")
        if p.startswith("datasets/"):
            return p[len("datasets/") :]
        return p

    for split in ["test", "val", "train"]:
        split_path = os.path.join(dataset_dir, dataset_name, f"{split}.csv")
        split_df = pd.read_csv(split_path)
        new_records = []
        for row in split_df.to_dict("records"):
            row["document_path"] = strip_path(row["document_path"])
            row["ocr"] = strip_path(row["ocr"])
            new_images = []
            for image in json.loads(row["image_files"]):
                new_images.append(strip_path(image))
            row["image_files"] = json.dumps(new_images)
            new_records.append(row)
        pd.DataFrame.from_records(new_records).to_csv(split_path)


def get_qa_dataset_info_table(dataset_dir="datasets"):
    files = glob.glob(os.path.join(dataset_dir, "*/qa.csv"))

    for f in files:
        df = pd.read_csv(f)
        print(f, Counter(df.error_type))


if __name__ == "__main__":
    fire.Fire(
        {
            "list_labels": list_labels,
            "generate_plots": generate_plots,
            "drop_label": drop_label_from_dataset,
            "cleanup_files": cleanup_files,
            "deduplicate": deduplicate,
            "remove_leading_path_names": remove_leading_path_names,
            "get_qa_dataset_info_table": get_qa_dataset_info_table,
            "rename_label": rename_label,
            "label_statistics": label_statistics,
        }
    )
