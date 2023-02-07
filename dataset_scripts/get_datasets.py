import glob
import pandas as pd
import traceback
import json
import fire
import os
import uuid

from sklearn.model_selection import train_test_split

from indico import IndicoClient, IndicoConfig
from indico.errors import IndicoError
from indico.queries import (
    RetrieveStorageObject,
    GetDataset,
    DownloadExport, CreateExport
)

JSON_TEMPLATE = (
    "indico-file:///storage/files/{dataset_id}/{file_id}_meta/page_info_{page}.json"
)

PNG_TEMPLATE = (
    "indico-file:///storage/files/{dataset_id}/{file_id}_meta/original_{page:05d}_page_{page}.png"
)

READAPI_PNG_TEMPLATE = (
    "indico-file:///storage/files/{dataset_id}/{file_id}_meta/original_page_{page}.png"
)

def get_export(client, dataset_id, labelset_id):
    # Get dataset object
    dataset = client.call(GetDataset(id=dataset_id))
    
    # Create export object using dataset's id and labelset id
    export = client.call(
        CreateExport(
            dataset_id=dataset.id,
            labelset_id=dataset.labelsets[0].id,
            file_info=True,
            wait=True
        )
    )

    # Use export object to download as pandas csv
    csv = client.call(DownloadExport(export.id))
    csv = csv.rename(columns=lambda col: col.rsplit("_", 1)[0])
    return csv

def reformat_labels(labels, document):
    print(labels)
    spans_labels = json.loads(labels)
    old_labels_i = []
    for target in spans_labels["targets"]:
        old_labels_i.append(
            {
                "label": target["label"],
                "start": min(l["start"] for l in target["spans"]),
                "end": max(l["end"] for l in target["spans"]),
            }
        )
        old_labels_i[-1]["text"] = document[
            old_labels_i[-1]["start"] : old_labels_i[-1]["end"]
        ]
    return json.dumps(old_labels_i)

def get_label_column(columns):
    for col in columns:
        if col not in ['Unnamed: 0', 'Indico_Id', 'document', 'file_id', 'file_name', 'file_url']:
            return col

def get_ocr(client, file_id, dataset_id):
    page_num = 0
    document = []
    while True:
        try:
            # JSONs are 0-indexed                                                                                                                      
            url = JSON_TEMPLATE.format(
                file_id=file_id, dataset_id=dataset_id, page=page_num
            )

            result = client.call(RetrieveStorageObject(url))
            document.append(result)
            page_num += 1
        except IndicoError:
            if len(document) == 0:
                print(f"possibly missing permissions for  dataset {dataset_id} retrying")
                raise
            traceback.print_exc()
            return document

def get_dataset(name, dataset_id, labelset_id, host="app.indico.io", api_token_path="prod_api_token.txt"):
    if not os.path.exists(name):
        os.mkdir(name)
        os.mkdir(os.path.join(name, "images"))
        os.mkdir(os.path.join(name, "files"))


    my_config = IndicoConfig(
        host=host, api_token_path=api_token_path,
    )
    client = IndicoClient(config=my_config)

    export_path = os.path.join(name, "raw_export.csv")

    if not os.path.exists(export_path):
        raw_export = get_export(client, dataset_id, labelset_id)
        raw_export.to_csv(export_path)
    else:
        raw_export = pd.read_csv(export_path)

    records = raw_export.to_dict("records")
    output_records = []
    for i, row in enumerate(records):
        label_col = get_label_column(row.keys())
        if pd.isna(row[label_col]):
            print("No labels - skipping")
            continue
        filename = str(uuid.uuid4())
        doc = get_ocr(client, row["file_id"], dataset_id)
        os.mkdir(os.path.join(name, "images", filename))
        page_pattern = os.path.join(name, "images", filename, "page_{}.png")
        output_record = {
            "ocr": json.dumps(doc),
            "text": row["document"],
            "labels": reformat_labels(row[label_col], row["document"]),
        }
        image_files = []
        for page in doc:
            page_image = page_pattern.format(page["pages"][0]["page_num"])
            image_files.append(page_image)
            with open(page_image, "wb") as fp:
                downloaded = False
                for template, offset in [(READAPI_PNG_TEMPLATE, 0), (PNG_TEMPLATE, 0)]:
                    print(
                        template.format(
                            page=page["pages"][0]["page_num"],
                            file_id=row["file_id"],
                            dataset_id=dataset_id
                        )
                    )
                    try:
                        fp.write(
                            client.call(
                                RetrieveStorageObject(
                                    template.format(
                                        page=page["pages"][0]["page_num"] + offset,
                                        file_id=row["file_id"],
                                        dataset_id=dataset_id
                                    )
                                )
                            )
                        )
                        downloaded = True
                        break
                    except Exception as e:
                        print(e)
                        pass
                if not downloaded:
                    raise Exception("Failed to download image")
        output_record["image_files"] = json.dumps(image_files)
        document_path = os.path.join(name, "files", filename + "." + row["file_name"].split(".")[-1])
        output_record["document_path"] = document_path
        with open(document_path, "wb") as fp:
            fp.write(client.call(RetrieveStorageObject(row["file_url"])))
        output_records.append(output_record)
    train_records, test_val_records = train_test_split(output_records, test_size=0.4)
    test_records, val_records = train_test_split(test_val_records, test_size=0.5)
    for split, records in [
            ("train", train_records),
            ("test", test_records),
            ("val", val_records),
    ]:
        pd.DataFrame.from_records(records).to_csv(os.path.join(name, "{}.csv".format(split)))
    os.remove(export_path)
        
    
if __name__ == "__main__":
    fire.Fire(get_dataset)
