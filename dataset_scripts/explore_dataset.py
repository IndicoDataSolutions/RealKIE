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
        assert len(expected) == len(set(expected)) # No duplicate files referenced
        all_files = set(glob.glob(os.path.join(dir, "*")))
        for e in expected:
            assert e in all_files, e
            all_files.remove(e)
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
            return p[len("datasets/"): ]
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
        }
    )
