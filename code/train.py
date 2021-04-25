import pickle as pickle
import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    BertConfig,
)
from transformers import (
    ElectraTokenizer,
    ElectraConfig,
    ElectraForSequenceClassification,
)
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    XLMRobertaConfig,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from load_data import *
import wandb
import random


def compute_metrics(pred):
    """
    평가를 위한 metrics function
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)  # % 기준으로 return
    wandb.log({"validation_accuracy": acc})
    return {
        "accuracy": acc,
    }


def set_seed(seed):
    # Set Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def set_config():
    wandb.login()
    CFG = wandb.config

    CFG.name = "xlm-roberta-large_preprocessing50_SEP_lr1e5_batch16_weight_decay1e7_warmup_steps1000_seed2021"
    CFG.seed = 2021
    CFG.MODEL_NAME = "xlm-roberta-large"
    CFG.learning_rate = 1e-5
    CFG.batch_size = 16
    CFG.epochs = 20
    CFG.tokenizer_max_length = 200
    CFG.warmup_steps = 1000
    CFG.weight_decay = 1e-7
    CFG.weight_decay = 0
    CFG.fp16 = True

    run = wandb.init(
        project="P-stage_2", group=CFG.MODEL_NAME, name=CFG.name, config=CFG
    )

    return CFG


def mytrain_wandb():
    # set config
    CFG = set_config()
    set_seed(CFG.seed)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)

    # load dataset
    train_dataset = load_data(
        "/opt/ml/input/data/train/train.tsv",
        truncation=True,
        token_highlighting_entity=False,
    )
    train_dataset, valid_dataset = train_valid_split(train_dataset)
    train_label = train_dataset["label"].values
    valid_label = valid_dataset["label"].values

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_valid = tokenized_dataset(valid_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # setting model hyperparameter
    config = AutoConfig.from_pretrained(CFG.MODEL_NAME)
    config.num_labels = 42
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.MODEL_NAME, config=config
    )
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        save_total_limit=5,  # number of total save model.
        save_steps=500,  # model saving step.
        num_train_epochs=CFG.epochs,  # total number of training epochs
        learning_rate=CFG.learning_rate,  # learning_rate
        per_device_train_batch_size=CFG.batch_size,  # batch size per device during training
        per_device_eval_batch_size=CFG.batch_size,  # batch size for evaluation
        warmup_steps=CFG.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=CFG.weight_decay,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=100,  # log saving step.
        evaluation_strategy="steps",  # evaluation strategy to adopt during training
        fp16=CFG.fp16,
        eval_steps=100,  # evaluation step.
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
    )

    # train model
    trainer.train()


def main():
    # mytrain()
    mytrain_wandb()


if __name__ == "__main__":
    main()
