import json
import os

import fire
import pandas as pd
import wandb
import tqdm


def fetch_all_runs(project_name, entity_name=None):
    wandb.login()
    runs = wandb.Api(timeout=30).runs(
        path=f"{entity_name}/{project_name}" if entity_name else project_name
    )
    return runs


def get_runs(
    project="dataset-paper-sweeps", entity="indico", cache_file="metrics.json"
):
    if os.path.exists(cache_file):
        with open(cache_file, "rt") as fp:
            return json.load(fp)
    all_runs = fetch_all_runs(project_name=project, entity_name=entity)
    metrics = []
    for a in tqdm.tqdm(all_runs, desc="downloading runs"):
        if a.state == "finished":
            try:
                metadata = json.load(a.file("wandb-metadata.json").download(replace=True))
                runtime = json.load(a.file("wandb-summary.json").download(replace=True))["_wandb"]["runtime"]
            except:
                print("Failed to download metadata and runtime")
                metadata = {}
                runtime = None
            metrics.append(
                {
                    "id": a.id,
                    "metrics": a.summaryMetrics,
                    "base_model": a.config["base_model"],
                    "training_lib": a.config["training_lib"],
                    "dataset": a.config["dataset_name"],
                    "config": a.config,
                    "metadata": metadata,
                    "runtime": runtime,
                }
            )

    for m in metrics:
        m["test_macro_f1"] = m["metrics"].get("test_macro_f1", 0.0)
        m["val_macro_f1"] = m["metrics"].get("val_macro_f1", 0.0)
    with open(cache_file, "wt") as fp:
        json.dump(metrics, fp)
    return metrics


def get_best_runs(project="dataset-paper-sweeps", entity="indico"):
    metrics_df = pd.DataFrame().from_records(get_runs(project=project, entity=entity))
    max_indices = metrics_df.groupby(["dataset", "base_model", "training_lib"])[
        "val_macro_f1"
    ].idxmax()
    max_val_macro_f1_rows = metrics_df.loc[max_indices]
    max_val_macro_f1_rows.reset_index(drop=True, inplace=True)
    return max_val_macro_f1_rows


def _format_metric(value):
    return f"{100 * value:.1f}"


def make_bold(value):
    return f"\\textbf{{{value}}}"


def format_metric(value, other_values):
    formatted_value = _format_metric(value)
    is_max = _format_metric(max(other_values)) == formatted_value
    if is_max:
        return make_bold(formatted_value), is_max
    else:
        return formatted_value, is_max


def format_param(value, name):
    if isinstance(value, str):
        if "_" in value:
            return value.replace("_", "\\_")
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, float):
        if float_e_format[name]:
            return f"{value:.1E}"
        return f"{value:.2f}"
    return str(value)


def get_param(config, keys):
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if key in config:
            return config[key]
    return ""


dataset_rename_dict = {
    "charities": "Charities",
    "fcc_invoices": "FCC Invoices",
    "nda": "NDA",
    "resource_contracts": "Resource Contracts",
    "s1": "S1",
    "s1_pages": "S1 (Pages)",
}

model_rename_dict = {
    "layoutlmv3": "LayoutLM V3 Base",
    "roberta-base": "RoBERTa Base",
    "microsoft/deberta-v3-base": "DeBERTa V3 Base",
    "allenai/longformer-base-4096": "Longformer Base",
    "xdoc": "XDoc Base",
}


headers = {
    "Auto Negative Sampling": "auto_negative_sampling",
    "Max Empty Chunk Ratio": ["max_empty_chunk_ratio", "empty_chunk_ratio"],
    "Learning Rate": ["learning_rate", "lr"],
    "Batch Size": ["batch_size", "per_device_train_batch_size"],
    "Num Epochs": ["n_epochs", "num_train_epochs"],
    "Class Weights": "class_weights",
    "LR Warmup": ["lr_warmup", "warmup_ratio"],
    "Collapse Whitespacee": "collapse_whitespace",
    "Max Grad Norm": "max_grad_norm",
    "L2 Regularization": ["l2_reg", "weight_decay"],
    "Gradient Accumulation Steps": "gradient_accumulation_steps",
    "LR Schedule": "lr_scheduler_type",
}

float_e_format = {
    "Learning Rate": True,
    "LR Warmup": False,
    "L2 Regularization": True,
    "Max Grad Norm": True,
    "Max Empty Chunk Ratio": False,
}


def create_summary_results_table(project="dataset-paper-sweeps", entity="indico"):
    max_val_macro_f1 = get_best_runs(project=project, entity=entity)
    latex_table = "\\begin{tabular}{|c|c|c|c|c|}\n"
    latex_table += "\\hline\n"
    latex_table += "\\textbf{Dataset} & \\textbf{Base Model} & \\textbf{Test Macro F1} & \\textbf{Val Macro F1} \\\\\n"
    latex_table += "\\hline\n"

    for dataset, group_data in max_val_macro_f1.groupby("dataset"):
        dataset_name = dataset_rename_dict.get(dataset, dataset)
        for i, (_, row) in enumerate(group_data.iterrows()):
            base_model_name = model_rename_dict.get(
                row["base_model"], row["base_model"]
            )
            if row["base_model"] == "roberta-base":
                base_model_name = f"{base_model_name} ({row['training_lib']})"
            if i == 0:
                latex_table += "\\hline"
                dataset_header = (
                    f"\\multirow{{{len(group_data)}}}{{*}}{{{dataset_name}}}"
                )
            else:
                dataset_header = ""
                latex_table += "\\cline{2-4}\n"
            test_value, is_best = format_metric(
                row["test_macro_f1"], group_data["test_macro_f1"]
            )
            val_value, _ = format_metric(
                row["val_macro_f1"], group_data["val_macro_f1"]
            )
            if is_best:
                base_model_name = make_bold(base_model_name)
            latex_table += f"{dataset_header} & {base_model_name} & {test_value} & {val_value} \\\\\n"

    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}"
    print(latex_table)


def create_sweep_param_results_table(project="dataset-paper-sweeps", entity="indico"):
    max_val_macro_f1 = get_best_runs(project=project, entity=entity)

    param_table_def = "|" + "|".join(["c" for _ in range(3 + len(headers))]) + "|"
    latex_sweep_table = (
        "\\begin{table*}\\centering\\scriptsize{\\begin{tabular}{"
        + param_table_def
        + "}\n"
    )
    latex_sweep_table += "\\hline\n"

    param_headers = " & ".join(
        make_bold(f"\\rot{{{param_name}}}") for param_name in headers.keys()
    )
    latex_sweep_table += (
        "\\textbf{Base Model} & \\textbf{Dataset} & \\textbf{F1} & "
        + param_headers
        + "\\\\\n"
    )
    latex_sweep_table += "\\hline\n"

    for (base_model, training_lib), group_data in max_val_macro_f1.groupby(
        ["base_model", "training_lib"]
    ):
        base_model_name = model_rename_dict.get(base_model, base_model)
        if base_model == "roberta-base":
            base_model_name = f"\shortstack{{{base_model_name}\\\\({training_lib})}}"

        for i, (_, row) in enumerate(group_data.iterrows()):
            dataset = row["dataset"]
            dataset_name = dataset_rename_dict.get(dataset, dataset)

            if i == 0:
                latex_sweep_table += "\\hline"
                base_model_header = (
                    f"\\multirow{{{len(group_data)}}}{{*}}{{{base_model_name}}}"
                )
            else:
                base_model_header = ""
                latex_sweep_table += f"\\cline{{2-{len(headers) + 3}}}\n"
            test_value = _format_metric(row["test_macro_f1"])
            print(row["config"].keys())
            param_values = " & ".join(
                format_param(get_param(row["config"], param_key), param_name)
                for param_name, param_key in headers.items()
            )
            latex_sweep_table += f"{base_model_header} & {dataset_name} & {test_value} & {param_values} \\\\\n"

    latex_sweep_table += "\\hline\n"
    latex_sweep_table += "\\end{tabular}}\\end{table*}"
    print(latex_sweep_table)


def get_compute_requirements(project="dataset-paper-sweeps", entity="indico"):
    kg_per_hour = {
        'Tesla T4': 0.021, #70W x 1h = 0.07 kWh x 0.3 kg eq. CO2/kWh - AWS US-west-2 according to https://mlco2.github.io/impact/#co2eq
        'NVIDIA GeForce RTX 2080 Ti': 0.0495, # 250W x 1h = 0.25 kWh x 0.198 kg eq. CO2/kWh = 0.05 kg eq. CO2 (198g / kWh is uk grid average 2021-2022)
        # The titan V machine also has 2 1080TI although these are the both 250W so the calculation is the same.
        'NVIDIA TITAN V': 0.0495, # 250W x 1h = 0.25 kWh x 0.198 kg eq. CO2/kWh = 0.05 kg eq. CO2 (198g / kWh is uk grid average 2021-2022)
    }

    all_runs = get_runs(project=project, entity=entity)
    records = []
    for run in all_runs:
        if run["metadata"]:
            records.append(
                {
                    "runtime_seconds": run["runtime"],
                    "runtime_hours": run["runtime"] / 3600,
                    "base_model": run["base_model"],
                    "training_lib": run["training_lib"],
                    "gpu": run["metadata"]["gpu"],
                    "kg_co2": run["runtime"] / 3600 * kg_per_hour[run["metadata"]["gpu"]]
                }
            )
    df = pd.DataFrame.from_records(records)
    # Group by columns and calculate totals
    grouped = df.groupby(["base_model"]).agg(
        count=("base_model", "count"),
        runtime_hours=("runtime_hours", "sum"),
        kg_co2=("kg_co2", "sum"),
    ).reset_index()

    # Calculate totals for all rows
    total_row = grouped[["count", "runtime_hours", "kg_co2"]].sum()
    total_row["base_model"] = "Total"

    # Append the total row to the grouped DataFrame
    grouped = grouped.append(total_row, ignore_index=True)

    # Convert DataFrame to LaTeX tabular format
    latex_table = grouped.to_latex(index=False, escape=False, column_format="c"*5)
    print(latex_table)



if __name__ == "__main__":
    fire.Fire(
        {
            "create_sweep_param_results_table": create_sweep_param_results_table,
            "create_summary_results_table": create_summary_results_table,
            "get_compute_requirements": get_compute_requirements
        }
    )
