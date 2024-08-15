import streamlit as st
import pandas as pd
import os
import glob
import json
import gzip
from PIL import Image, ImageDraw
import random


def sequences_overlap(x: dict, y: dict) -> bool:
    return x["start"] < y["end"] and y["start"] < x["end"]


def draw_page(image_path, page_number, ocr_path, labels, label_colors):
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    with gzip.open(ocr_path) as fp:
        page_ocr = json.load(fp)[page_number]
    page_labels = [
        l for l in labels if sequences_overlap(page_ocr["pages"][0]["doc_offset"], l)
    ]
    for label in page_labels:
        print(page_ocr["chars"][0])
        labeled_chars = [
            c
            for c in page_ocr["chars"]
            if sequences_overlap(
                {"start": c["doc_index"], "end": c["doc_index"] + 1}, label
            )
        ]
        color = label_colors.get(
            label["label"], (255, 255, 255)
        ) 
        for char in labeled_chars:
            box = char["position"]
            top_left = (box["left"], box["top"])
            bottom_right = (box["right"], box["bottom"])

            draw.rectangle(
                [top_left, bottom_right], fill=color + (128,)
            )

    combined = Image.alpha_composite(image, overlay)

    return combined.convert(
        "RGB"
    )  


def load_datasets(base_dir):
    datasets = glob.glob(os.path.join(base_dir, "*"))
    return [os.path.basename(d) for d in datasets]


def load_split(dataset_name, split_name, base_dir):
    csv_path = os.path.join(base_dir, dataset_name, f"{split_name}.csv")
    return pd.read_csv(csv_path)


def extract_labels(dataset_name, base_dir):
    labels_set = set()
    for split in ["train", "test", "val"]:
        df = load_split(dataset_name, split, base_dir=base_dir)
        for label_list in df["labels"]:
            labels = json.loads(label_list)
            for label in labels:
                labels_set.add(label["label"])
    return sorted(labels_set)


def generate_label_colors(labels_set):
    random.seed(42)
    label_colors = {}
    for label in labels_set:
        label_colors[label] = tuple(random.randint(0, 255) for _ in range(3))
    return label_colors


def app(base_dir="datasets"):
    st.title("Document Page Viewer")
    datasets = load_datasets(base_dir=base_dir)
    dataset_name = st.selectbox("Select Dataset", datasets)
    split = st.selectbox("Select Split", ["train", "test", "val"])
    df = load_split(dataset_name, split, base_dir=base_dir)

    labels_set = extract_labels(dataset_name, base_dir=base_dir)
    label_colors = generate_label_colors(labels_set)

    st.subheader("Label Key")
    for label, color in label_colors.items():
        st.markdown(
            f"<div style='display: flex; align-items: center;'>"
            f"<div style='width: 20px; height: 20px; background-color: rgb{color}; margin-right: 10px;'></div>"
            f"{label}</div>",
            unsafe_allow_html=True,
        )
    filename = st.selectbox("Select File", df["original_filename"].unique())
    selected_row = df[df["original_filename"] == filename].iloc[0]
    image_files = json.loads(selected_row["image_files"])
    ocr_path = selected_row["ocr"]
    labels = json.loads(selected_row["labels"])
    if len(image_files) > 1:
        page_number = st.slider("Select Page", 0, len(image_files) - 1)
    else:
        page_number = 0
    image_path = image_files[page_number]
    image = draw_page(
        os.path.join(base_dir, image_path),
        page_number,
        os.path.join(base_dir, ocr_path),
        labels=labels,
        label_colors=label_colors,
    )
    st.subheader("Document Page")
    st.image(image, caption=f"Page {page_number} of {filename}")


if __name__ == "__main__":
    app()
