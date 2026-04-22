from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, SentenceTransformerTrainingArguments, evaluation

from datasets import Dataset

import pandas as pd

SEED = 24

model_name = "Lajavaness/sentence-camembert-base"
#model_name = "./sbert-parlement-2/checkpoint-1000"
model = SentenceTransformer(
    model_name,
    device="cuda"
)

dataset = pd.read_csv("./dataset/dataset.csv", dtype={"agreement": "float64"})
dataset = dataset[["a_speech", "b_speech", "agreement"]]
dataset = dataset.rename(columns={"agreement":"score"})

train, test = train_test_split(dataset, test_size=0.2, random_state=SEED, stratify=dataset["score"])
train, dev = train_test_split(train, test_size=0.1, random_state=SEED, stratify=train["score"])

train, test, dev = Dataset.from_pandas(train, preserve_index=False), Dataset.from_pandas(test, preserve_index=False), Dataset.from_pandas(dev, preserve_index=False)

loss = losses.CoSENTLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir="sbert-parlement-cosentloss-4",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_steps=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
)

evaluator = evaluation.EmbeddingSimilarityEvaluator(
    sentences1=dev["a_speech"],
    sentences2=dev["b_speech"],
    scores=dev["score"],
    main_similarity=evaluation.SimilarityFunction.COSINE,
    name="agreement-eval"
)

trainer = SentenceTransformerTrainer(
    args=args,
    model=model,
    train_dataset=train,
    eval_dataset=dev,
    loss=loss,
    evaluator=evaluator
)

trainer.train()