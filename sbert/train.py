from sklearn.model_selection import train_test_split
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import (
    ContrastiveLoss,
    SiameseDistanceMetric,
    TripletLoss,
    TripletDistanceMetric,
)
from sentence_transformers.sentence_transformer.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
    TripletEvaluator,
    SequentialEvaluator
)

from peft import LoraConfig, TaskType

from datasets import Dataset

import pandas as pd


SEED = 24
MARGIN_LOSS_SIAMESE = 0.2
MARGIN_LOSS_TRIPLET = 0.2
DISTANCE_METRIC_SIAMESE = SiameseDistanceMetric.COSINE_DISTANCE
DISTANCE_METRIC_TRIPLET = TripletDistanceMetric.COSINE

model_name = "medialab-sciencespo/ParlBERT"
model = SentenceTransformer(model_name, device="cuda")
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    target_modules=["query", "key", "value", "dense"],
    r=16,
    lora_alpha=64,
    lora_dropout=0.1
)
model.add_adapter(peft_config)

# ==== DATASETS ====

# Loading triplet dataset

dataset_triplet = pd.read_csv("./dataset/triplet.csv", dtype={"agreement": "float64"})
dataset_triplet = dataset_triplet[["anc_speech", "pos_speech", "neg_speech"]]

train_triplet, test_triplet = train_test_split(
    dataset_triplet, test_size=0.2, random_state=SEED
)
train_triplet, dev_triplet = train_test_split(
    train_triplet, test_size=0.1, random_state=SEED
)

train_triplet, test_triplet, dev_triplet = (
    Dataset.from_pandas(train_triplet, preserve_index=False),
    Dataset.from_pandas(test_triplet, preserve_index=False),
    Dataset.from_pandas(dev_triplet, preserve_index=False),
)

# Loading duplet dataset

dataset_duplet = pd.read_csv("./dataset/duplet.csv", dtype={"agreement": "float64"})
dataset_duplet = dataset_duplet[["a_speech", "b_speech", "agreement"]]
dataset_duplet = dataset_duplet.rename(columns={"agreement": "score"})

train_duplet, test_duplet = train_test_split(
    dataset_duplet, test_size=0.2, random_state=SEED, stratify=dataset_duplet["score"]
)
train_duplet, dev_duplet = train_test_split(
    train_duplet, test_size=0.1, random_state=SEED, stratify=train_duplet["score"]
)

train_duplet, test_duplet, dev_duplet = (
    Dataset.from_pandas(train_duplet, preserve_index=False),
    Dataset.from_pandas(test_duplet, preserve_index=False),
    Dataset.from_pandas(dev_duplet, preserve_index=False),
)

train_dataset = {
    "duplet": train_duplet,
    "triplet": train_triplet
}

dev_dataset = {
    "duplet": dev_duplet,
    "triplet": dev_triplet
}

# ==== LOSSES ====

losses = {
    "duplet": ContrastiveLoss(model, margin=MARGIN_LOSS_SIAMESE, distance_metric=DISTANCE_METRIC_SIAMESE),
    "triplet": TripletLoss(model, triplet_margin=MARGIN_LOSS_TRIPLET, distance_metric=DISTANCE_METRIC_TRIPLET)
}


# ==== TRAINING ====

args = SentenceTransformerTrainingArguments(
    output_dir="checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
    use_cache=False,
    gradient_checkpointing=True,
    gradient_accumulation_steps=8
)

# ==== EVALUATORS ====

evaluator_duplet = EmbeddingSimilarityEvaluator(
    sentences1=dev_duplet["a_speech"],
    sentences2=dev_duplet["b_speech"],
    scores=dev_duplet["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="agreement-eval-duplet",
)

evaluator_triplet = TripletEvaluator(
    anchors=dev_triplet["anc_speech"],
    positives=dev_triplet["pos_speech"],
    negatives=dev_triplet["neg_speech"],
    main_similarity_function=SimilarityFunction.COSINE,
    name="agreement-eval-triplet",
)

trainer = SentenceTransformerTrainer(
    args=args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    loss=losses,
    evaluator=SequentialEvaluator([evaluator_triplet, evaluator_duplet]),
)

trainer.train()

model.save_pretrained("./adapter")