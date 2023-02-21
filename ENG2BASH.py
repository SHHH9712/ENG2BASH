import transformers
from huggingface_hub import notebook_login, login
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
import numpy as np
import os
import pathlib

max_input_length = 128
max_target_length = 128

model_checkpoint = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
metric = load_metric("sacrebleu")


def load_csv(file_path):
    # Input: csv_file_name
    # Output: pandas data frame
    df = pd.read_csv(file_path)
    input_df, output_df = df["input"], df["output"]
    return input_df, output_df


def preprocess_function(examples):
    inputs = [ex for ex in examples["input"]]
    targets = [ex for ex in examples["output"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def main():

    input, output = load_csv(
        os.path.join(pathlib.Path(__file__).parent.resolve(), "customv2.csv")
    )

    print(transformers.__version__)
    notebook_login()

    # x_train, x_test, y_train, y_test = train_test_split(
    #     input, output, train_size=0.8, test_size=0.2, shuffle=True, random_state=0
    # )
    # x_val, x_test, y_val, y_test = train_test_split(
    #     x_test, y_test, train_size=0.5, test_size=0.5, shuffle=True, random_state=0
    # )

    # train = pd.concat([x_train, y_train], axis=1)
    # test = pd.concat([x_test, y_test], axis=1)
    # valid = pd.concat([x_val, y_val], axis=1)

    # train_dataset = Dataset.from_pandas(train)
    # train_dataset = train_dataset.remove_columns(["__index_level_0__"])

    # test_dataset = Dataset.from_pandas(test)
    # test_dataset = test_dataset.remove_columns(["__index_level_0__"])

    # valid_dataset = Dataset.from_pandas(valid)
    # valid_dataset = valid_dataset.remove_columns(["__index_level_0__"])

    # raw_datasets = DatasetDict(
    #     {"train": train_dataset, "test": test_dataset, "valid": valid_dataset}
    # )
    # raw_datasets["test"][1]

    # tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    # batch_size = 5
    # model_name = model_checkpoint.split("/")[-1]
    # args = Seq2SeqTrainingArguments(
    #     f"{model_name}-transferLearning-NLP2BASH",
    #     evaluation_strategy="epoch",
    #     learning_rate=2e-4,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     weight_decay=0.01,
    #     save_total_limit=3,
    #     num_train_epochs=5,
    #     predict_with_generate=True,
    #     fp16=True,
    #     push_to_hub=True,
    # )

    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # trainer = Seq2SeqTrainer(
    #     model,
    #     args,
    #     train_dataset=tokenized_datasets["train"],
    #     eval_dataset=tokenized_datasets["valid"],
    #     data_collator=data_collator,
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics,
    # )
    # trainer.train()
    # trainer.push_to_hub()
