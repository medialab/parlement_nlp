import argparse
import optuna

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

from urllib.parse import urlencode

SEED = 42

def main():
    parser = argparse.ArgumentParser(description='BERT adaptation script')
    parser.add_argument('--dataset', type=str, required=True, help='Path to BERT training dataset')
    
    args = parser.parse_args()
    input_path = args.dataset

    model_name = "almanach/camembertv2-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset("csv", data_files={"train": input_path}, split="train")

    accuracy = evaluate.load("accuracy")

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
        return accuracy.compute(predictions=filtered_preds, references=filtered_labels)
    
    def compute_objectives(metrics):
        return metrics["eval_accuracy"]

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.train_test_split(test_size=0.1)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, seed=SEED)

    model = AutoModelForMaskedLM.from_pretrained(model_name, dtype="auto")

    args = TrainingArguments(
        output_dir="output",
        # training and duration
        # learning rate and scheduler
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

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [2e-5, 5e-5, 2e-4]
            ),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [8, 16, 32]
            ),
            "num_train_epochs": trial.suggest_int(
                "num_train_epochs", 3, 10
            ),
        }
    

    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=5,
        compute_objective=compute_objectives,
        study_name="parlbert_optuna",
        storage="sqlite:///optuna.db",
        load_if_exists=True
    )
    
    print(best_run)

if __name__ == '__main__':
    main()