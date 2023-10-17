import logging
import os
import fire

import wandb

from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


from hf_helpers import HFLoader
from helpers import (
    get_matching_sweep,
    run_agent,
    get_dataset,
)
from all_sweep_configs import get_configs_for_model

logger = logging.getLogger(__name__)

TRAINING_LIB = "huggingface"


def train_and_predict(
    dataset_name,
    dataset_dir="/datasets",
    base_model="distilbert-base-uncased",
    output_dir=None,
    preprocessing_num_workers=None,
    overwrite_cache=True,
    empty_chunk_ratio=2.0,
    per_device_train_batch_size=2,
    num_train_epochs=16,
    gradient_accumulation_steps=1,
    **training_args,
):
    if output_dir is None:
        output_dir = os.path.join("outputs", dataset_name, "model_output")

    train_dataset = get_dataset(dataset_name, "train", dataset_dir)

    loader = HFLoader(
        dataset_name,
        dataset_dir,
        base_model,
        preprocessing_num_workers,
        overwrite_cache,
        empty_chunk_ratio,
    )
    train_dataset = loader.get_hf_dataset("train", is_training=True)
    eval_dataset = loader.get_hf_dataset("val")
    data_collator = DataCollatorForTokenClassification(
        tokenizer=loader.tokenize_and_align_labels
    )

    trainer = Trainer(
        model=loader.model,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            save_total_limit=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **training_args,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=loader.tokenizer,
        data_collator=data_collator,
        compute_metrics=loader.compute_metrics,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Prediction
    logger.info("*** Predict ***")

    for split in ["train", "val", "test"]:
        loader.run_predictions(
            trainer,
            split,
            output_path=os.path.join(output_dir, f"{split}_predictions.csv"),
        )


def setup_and_run_sweep(
    project, entity, dataset_name, base_model, dataset_dir="/datasets"
):
    sweep_id_config = {
        "dataset_name": {"value": dataset_name},
        "dataset_dir": {"value": dataset_dir},
        "base_model": {"value": base_model},
        "training_lib": {"value": TRAINING_LIB},
    }
    sweep_id = get_matching_sweep(project, entity, sweep_id_config)
    if sweep_id is not None:
        print(f"Resuming Sweep with ID: {sweep_id}")
        return run_agent(
            sweep_id,
            entity=entity,
            project=project,
            training_lib=TRAINING_LIB,
            train_and_predict=train_and_predict,
        )
    sweep_configs = get_configs_for_model(TRAINING_LIB, sweep_id_config)
    sweep_id = wandb.sweep(sweep_configs, project=project, entity=entity)
    print(f"Your sweep id is {sweep_id}")
    return run_agent(
        sweep_id,
        entity=entity,
        project=project,
        training_lib=TRAINING_LIB,
        train_and_predict=train_and_predict,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "train_and_predict": train_and_predict,
            "wandb_sweep": setup_and_run_sweep,
            "wandb_resume_sweep": run_agent,
        }
    )
