from sklearn.model_selection import train_test_split
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import (
    ContrastiveLoss,
    SiameseDistanceMetric,
)
from sentence_transformers.sentence_transformer.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)

from datasets import Dataset

import pandas as pd

SEED = 24
MARGIN_LOSS = 0.5
DISTANCE_METRIC = SiameseDistanceMetric.COSINE_DISTANCE

model_name = "./model-triplet"
model = SentenceTransformer(model_name, device="cuda")

dataset = pd.read_csv("./dataset/duplet.csv", dtype={"agreement": "float64"})
dataset = dataset[["a_speech", "b_speech", "agreement"]]
dataset = dataset.rename(columns={"agreement": "score"})

train, test = train_test_split(
    dataset, test_size=0.2, random_state=SEED, stratify=dataset["score"]
)
train, dev = train_test_split(
    train, test_size=0.1, random_state=SEED, stratify=train["score"]
)

train, test, dev = (
    Dataset.from_pandas(train, preserve_index=False),
    Dataset.from_pandas(test, preserve_index=False),
    Dataset.from_pandas(dev, preserve_index=False),
)

loss = ContrastiveLoss(model, margin=MARGIN_LOSS, distance_metric=DISTANCE_METRIC)

args = SentenceTransformerTrainingArguments(
    output_dir="duplet",
    num_train_epochs=10,
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

evaluator = EmbeddingSimilarityEvaluator(
    sentences1=dev["a_speech"],
    sentences2=dev["b_speech"],
    scores=dev["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="agreement-eval",
)

trainer = SentenceTransformerTrainer(
    args=args,
    model=model,
    train_dataset=train,
    eval_dataset=dev,
    loss=loss,
    evaluator=evaluator,
)

trainer.train()

trainer.save_model("./model-duplet")