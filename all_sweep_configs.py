FINETUNE_SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_macro_f1", "goal": "maximize"},
    "parameters": {
        "auto_negative_sampling": {"values": [False, True]},
        "max_empty_chunk_ratio": {
            "distribution": "log_uniform_values",
            "min": 1e-2,
            "max": 1000.0,
        },
        "lr": {"distribution": "log_uniform_values", "min": 1e-8, "max": 1e-2},
        "batch_size": {"distribution": "int_uniform", "min": 1, "max": 8},
        "n_epochs": {"distribution": "int_uniform", "min": 1, "max": 16},
        "class_weights": {"values": [None, "linear", "sqrt", "log"]},
        "lr_warmup": {"distribution": "uniform", "min": 0, "max": 0.5},
        "collapse_whitespace": {"values": [True, False]},
        "max_grad_norm": {
            "distribution": "log_uniform_values",
            "min": 1e-3,
            "max": 1e5,
        },
        "l2_reg": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1.0},
    },
}

LAYOUTLM_SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_macro_f1", "goal": "maximize"},
    "parameters": {
        "empty_chunk_ratio": {
            "distribution": "log_uniform_values",
            "min": 1e-2,
            "max": 1000.0,
        },
        "num_train_epochs": {"distribution": "int_uniform", "min": 1, "max": 64},
        "per_device_train_batch_size": {
            "distribution": "int_uniform",
            "min": 1,
            "max": 2,
        },
        "gradient_accumulation_steps": {
            "distribution": "int_uniform",
            "min": 1,
            "max": 8,
        },
        "warmup_ratio": {"distribution": "uniform", "min": 0, "max": 0.5},
        "warmup_steps": {"value": 0},  # To ensure precedence goes to warmup_ratio.
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-8,
            "max": 1e-2,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1.0,
        },
        "max_grad_norm": {
            "distribution": "log_uniform_values",
            "min": 1e-3,
            "max": 1e5,
        },
        "lr_scheduler_type": {
            "values": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "constant",
                "constant_with_warmup",
                "inverse_sqrt",
            ]
        },
    },
}

HUGGINGFACE_SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_macro_f1", "goal": "maximize"},
    "parameters": {
        "empty_chunk_ratio": {
            "distribution": "log_uniform_values",
            "min": 1e-2,
            "max": 1000.0,
        },
        "num_train_epochs": {"distribution": "int_uniform", "min": 1, "max": 16},
        "per_device_train_batch_size": {
            "distribution": "int_uniform",
            "min": 1,
            "max": 4,
        },
        "per_device_eval_batch_size": {
            "value": 2,
        },
        "gradient_accumulation_steps": {
            "distribution": "int_uniform",
            "min": 1,
            "max": 8,
        },
        "warmup_ratio": {"distribution": "uniform", "min": 0, "max": 0.5},
        "warmup_steps": {"value": 0},  # To ensure precedence goes to warmup_ratio.
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-8,
            "max": 1e-2,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1.0,
        },
        "max_grad_norm": {
            "distribution": "log_uniform_values",
            "min": 1e-3,
            "max": 1e5,
        },
        "lr_scheduler_type": {
            "values": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "constant",
                "constant_with_warmup",
                "inverse_sqrt",
            ]
        },
    },
}


def get_configs_for_model(model, sweep_id_config):
    if model == "finetune":
        config_dict = FINETUNE_SWEEP_CONFIG
    elif model == "hf":
        config_dict = HUGGINGFACE_SWEEP_CONFIG
    elif model == "layoutlm":
        config_dict = LAYOUTLM_SWEEP_CONFIG
    else:
        raise ValueError(f"Model {model} not recognized.")

    config_dict["parameters"].update(sweep_id_config)
    return config_dict
