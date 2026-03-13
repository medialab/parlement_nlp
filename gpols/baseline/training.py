"""
Script to fine-tune CamembertV2 on our motions ("amendements") and speechs

Cf. https://huggingface.co/docs/transformers/training?training-args=training+duration
"""
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    DataCollatorForLanguageModeling, 
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer
)
import numpy as np
import evaluate

SEED = 42

model_name = "almanach/camembertv2-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("csv", data_files={"train": "./data/dataset-camembert.csv"}, split="train")

accuracy = evaluate.load("accuracy")
perplexity = evaluate.load("perplexity")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
    )

# Cf. https://huggingface.co/docs/evaluate/transformers_integrations#trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    mask = labels != -100
    filtered_preds = predictions[mask]
    filtered_labels = predictions[mask]
    acc = accuracy.compute(predictions=filtered_preds, references=filtered_labels)

    return {
        "accuracy": acc["accuracy"],
    }

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.1)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, seed=SEED)

model = AutoModelForMaskedLM.from_pretrained(model_name, dtype="auto")

args = TrainingArguments(
    output_dir="camembertav2-finetuned",
    # training and duration
    num_train_epochs=3,
    per_device_train_batch_size=8,
    # learning rate and scheduler
    learning_rate=5e-5,
    warmup_steps=0.224,
    # regularization and training stability (defaults)
    # optimizer (defaults)
    # mixed precision training (defaults)
    # gradient checkpointing
    gradient_checkpointing=True,
    # logging
    logging_strategy="steps",
    logging_steps=100,
    # eval
    eval_strategy="steps",
    eval_steps=500,
    # save
    save_strategy="steps",
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()