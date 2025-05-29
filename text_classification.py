"""
Fine-tuning de los modelos de HuggingFace para text classification.
"""
# Archivo adaptado de https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py 
# (29 de mayo de 2025)
# Preparado para correr con Python 3.10

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.51.3")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

# Set variables
# General
account_hf = "TU CUENTA DE HUGGINGFACE" # en mi caso es "jorgeortizfuentes"
task_name = "fake-news"
last_checkpoint = None # Directorio del último checkpoint si quiero retomar un entrenamiento
token = True # Para usar datasets privados de HuggingFace y subir el modelo a HuggingFace Hub
push_to_hub = True # Para subir el modelo a HuggingFace Hub
do_train = True
do_eval = True
do_predict = True
output_predict_file = "predictions.txt"
evaluate_metric = "f1"
resume_from_checkpoint = None
language = "es"
overwrite_output_dir = True
seed = 13

# Datasets variables
dataset_name = f"{account_hf}/es_fakenews_dataset"
dataset_config_name = None
pad_to_max_length = True
max_train_samples = None # Por ej 1000 si no quiero entrenar con todo el dataset de training (sirve para pruebas)
max_eval_samples = None # si no quiero evaluar con todo el dataset de evaluación
max_predict_samples = None # si no quiero predecir con todo el dataset de testing

# Task and models
learning_rate = 2e-5
lr_scheduler_type = "linear"
auto_find_batch_size = True
per_device_train_batch_size = 8
per_device_eval_batch_size = 8
max_seq_length = 512
optim = "adamw_torch"
weight_decay = 0.01
num_train_epochs = 2
save_total_limit = 2
is_regression = False
model_name_or_path = "dccuchile/bert-base-spanish-wwm-cased"
use_fast_tokenizer = False
ignore_mismatched_sizes = True
fp16 = False
output_dir = f"models/{task_name}-{model_name_or_path.split('/')[-1]}"
evaluation_strategy = "epoch" # or "steps"
save_strategy = "epoch" # or "steps"
load_best_model_at_end = True
save_safetensors = True



# Detecting last checkpoint.
if os.path.isdir(output_dir) and do_train and not overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
        raise ValueError(
            f"Output directory ({output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None and resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

# Set seed before initializing model.
set_seed(seed)

# Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
# or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
#
# For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
# sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
# label if at least two columns are provided.
#
# If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
# single column. You can easily tweak this behavior (see below)
#
# In distributed training, the load_dataset function guarantee that only one local process can concurrently
# download the dataset.

# Downloading and loading a dataset from the hub.
raw_datasets = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=".hf_cache",
            token=True if token else None,
)


# Labels
if is_regression:
    num_labels = 1
else:
    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    label_list = raw_datasets["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

# Load pretrained model and tokenizer
#
# In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    finetuning_task=task_name,
    token=True if token else None,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    use_fast=use_fast_tokenizer,
    token=True if token else None,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    from_tf=bool(".ckpt" in model_name_or_path),
    token=True if token else None,
    ignore_mismatched_sizes=ignore_mismatched_sizes,
)


# Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
    sentence1_key, sentence2_key = "sentence1", "sentence2"
else:
    if len(non_label_column_names) >= 2:
        sentence1_key, sentence2_key = non_label_column_names[:2]
    else:
        sentence1_key, sentence2_key = non_label_column_names[0], None

# Padding strategy
if pad_to_max_length:
    padding = "max_length"
else:
    # We will pad later, dynamically at batch creation, to the max sequence length in each batch
    padding = False

# Some models have set the order of the labels to use, so let's make sure we do use it.
label_to_id = None
if (
    model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
    #and data_args.task_name is not None
    and not is_regression
):
    # Some have all caps in their config, some don't.
    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    if sorted(label_name_to_id.keys()) == sorted(label_list):
        label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    else:
        logger.warning(
            "Your model seems to have been trained with labels, but they don't match the dataset: ",
            f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
            "\nIgnoring the model labels as a result.",
        )

if label_to_id is not None:
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

if max_seq_length > tokenizer.model_max_length:
    logger.warning(
        f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
max_seq_length = min(max_seq_length, tokenizer.model_max_length)

def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    return result

# Preprocess the dataset
raw_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    #load_from_cache_file=not data_args.overwrite_cache,
    desc="Running tokenizer on dataset",
)
if do_train:
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if max_train_samples is not None:
        max_train_samples = min(len(train_dataset), max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

if do_eval:
    if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation"]
    if max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

if do_predict:
    if "test" not in raw_datasets and "test_matched" not in raw_datasets:
        raise ValueError("--do_predict requires a test dataset")
    predict_dataset = raw_datasets["test"]
    if max_predict_samples is not None:
        max_predict_samples = min(len(predict_dataset), max_predict_samples)
        predict_dataset = predict_dataset.select(range(max_predict_samples))

# Log a few random samples from the training set:
if do_train:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

# Get the metric function
if evaluate_metric is not None:
    metric = evaluate.load(evaluate_metric)
else:
    if is_regression:
        metric = evaluate.load("mse")
    else:
        metric = evaluate.load("accuracy")

# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result

# Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
# we already did the padding.
if pad_to_max_length:
    data_collator = default_data_collator
elif fp16:
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
else:
    data_collator = None

training_args = TrainingArguments(output_dir=output_dir, 
                                  overwrite_output_dir=overwrite_output_dir,
                                    eval_strategy=evaluation_strategy, 
                                    save_strategy=save_strategy,
                                    load_best_model_at_end=load_best_model_at_end,
                                    learning_rate=learning_rate, 
                                    lr_scheduler_type=lr_scheduler_type,
                                    per_device_train_batch_size=per_device_train_batch_size,
                                    per_device_eval_batch_size=per_device_eval_batch_size,
                                    auto_find_batch_size=True,
                                    num_train_epochs=num_train_epochs, 
                                    weight_decay=weight_decay, 
                                    fp16=fp16,
                                    metric_for_best_model=f"eval_{evaluate_metric}",
                                    seed=seed,
                                    data_seed=seed,
                                    optim=optim,
                                    save_total_limit=save_total_limit,
                                    save_safetensors=save_safetensors,
                                    hub_strategy="end" if push_to_hub else None,
                                    push_to_hub = push_to_hub,
                                    logging_strategy = "epoch",
                                    report_to="none",
                                    )      


# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if do_train else None,
    eval_dataset=eval_dataset if do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Training
if do_train:
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    max_train_samples = (
        max_train_samples if max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

# Evaluation
if training_args.do_eval:
    logger.info("*** Evaluate ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)

    metrics = trainer.evaluate(eval_dataset=eval_dataset)

    max_eval_samples = (
        max_eval_samples if max_eval_samples is not None else len(eval_dataset)
    )
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)

if do_predict:
    logger.info("*** Predict ***")


    # Removing the `label` columns because it contains -1 and Trainer won't like that.
    predict_dataset = predict_dataset.remove_columns("label")
    predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
    predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

    output_predict_file = os.path.join(output_dir, f"predict_results_{task_name}.txt")
    if trainer.is_world_process_zero():
        with open(output_predict_file, "w") as writer:
            logger.info(f"***** Predict results {task_name} *****")
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                if is_regression:
                    writer.write(f"{index}\t{item:3.3f}\n")
                else:
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")

kwargs = {"finetuned_from": model_name_or_path, "tasks": "text-classification"}
kwargs["language"] = language
if dataset_name is not None:
    kwargs["dataset_tags"] = dataset_name
    if dataset_config_name is not None:
        kwargs["dataset_args"] = dataset_config_name
        kwargs["dataset"] = f"{dataset_name} {dataset_config_name}"
    else:
        kwargs["dataset"] = dataset_name
if push_to_hub:
    trainer.push_to_hub(**kwargs)
else:
    trainer.create_model_card(**kwargs)