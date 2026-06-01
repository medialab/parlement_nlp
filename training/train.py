import logging
import csv
import os
import numpy as np

from scipy.special import kl_div
from sklearn.model_selection import train_test_split
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.util import pairwise_cos_sim
from sentence_transformers.sentence_transformer.losses import (
    ContrastiveLoss,
    SiameseDistanceMetric,
    TripletLoss,
    TripletDistanceMetric,
)
from sentence_transformers.sentence_transformer.evaluation import (
    BaseEvaluator,
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
    SequentialEvaluator,
)

from peft import LoraConfig, TaskType

from datasets import Dataset

import pandas as pd

import datetime
import time

logger = logging.getLogger(__name__)


class KLDivergenceEvaluator(BaseEvaluator):
    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        scores: list[float],
        batch_size: int = 16,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ):
        super().__init__()
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.write_csv = write_csv

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(scores)

        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file = (
            "kl_divergence_evaluation" + ("_" + name if name else "_") + "_results.csv"
        )

        self.csv_headers = ["epoch", "steps", "cosine_kl_div"]

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        logger.info(
            f"KLDivergenceEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:"
        )

        embeddings1 = self.embed_inputs(model, self.sentences1)
        embeddings2 = self.embed_inputs(model, self.sentences2)

        similarities = pairwise_cos_sim(embeddings1, embeddings2).detach().cpu().numpy()
        scores = np.array(self.scores)

        sim_0 = similarities[scores == 0.0]
        sim_1 = similarities[scores == 1.0]
        num_bins = 50
        bins = np.linspace(-1, 1, num_bins + 1)

        counts_0, _ = np.histogram(sim_0, bins=bins)
        counts_1, _ = np.histogram(sim_1, bins=bins)

        p = counts_0 / np.sum(counts_0)
        q = counts_1 / np.sum(counts_1)

        kl_all = kl_div(p, q)
        kl = np.nansum(kl_all)

        metrics = {"kl_div_cosine": kl}
        logger.info(f"COSINE-KL-Divergence: {kl:.4f}")

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(
                csv_path,
                newline="",
                mode="a" if output_file_exists else "w",
                encoding="utf-8",
            ) as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [epoch, steps, metrics["kl_div_cosine"]]
                )

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    def embed_inputs(
        self,
        model: SentenceTransformer,
        sentences: str | list[str] | np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        return model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            **kwargs,
        )

    @property
    def description(self) -> str:
        return "KL Divergence"


SEED = 24
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
MARGIN_LOSS_SIAMESE = 0.4
MARGIN_LOSS_TRIPLET = 0.4
DISTANCE_METRIC_SIAMESE = SiameseDistanceMetric.COSINE_DISTANCE
DISTANCE_METRIC_TRIPLET = TripletDistanceMetric.COSINE

cmd = datetime.datetime(2026, 5, 29, 21, 0)

while datetime.datetime.now() < cmd:
    print('waiting...', flush=True)
    time.sleep(10)


modules = {
    "Qwen/Qwen3-Embedding-0.6B": ["q_proj", "k_proj", "v_proj"],
    "Lajavaness/sentence-camembert-large": ["quey", "key", "value"]
}

model_name = "Qwen/Qwen3-Embedding-0.6B"
#model_name = "Lajavaness/sentence-camembert-large"
model = SentenceTransformer(model_name, device="cuda")
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    target_modules=modules[model_name],
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
)
model.add_adapter(peft_config)

# ==== DATASETS ====

# Loading triplet dataset

dataset_triplet = pd.read_csv("./data/triplets.csv", dtype={"agreement": "float64"})
dataset_triplet = dataset_triplet[["anc_speech", "pos_speech", "neg_speech"]]

train_triplet, test_triplet = train_test_split(
    dataset_triplet, test_size=0.2, random_state=SEED
)
train_triplet, dev_triplet = train_test_split(
    train_triplet, test_size=0.1, random_state=SEED
)

train_triplet, test_triplet = (
    Dataset.from_pandas(train_triplet, preserve_index=False),
    Dataset.from_pandas(test_triplet, preserve_index=False),
)

# Loading pair dataset

dataset_pair = pd.read_csv("./data/paires.csv", dtype={"agreement": "float64"})
dataset_pair = dataset_pair[["a_speech", "b_speech", "agreement"]]
dataset_pair = dataset_pair.rename(columns={"agreement": "score"})

train_pair, test_pair = train_test_split(
    dataset_pair, test_size=0.2, random_state=SEED, stratify=dataset_pair["score"]
)
train_pair, dev_pair = train_test_split(
    train_pair, test_size=0.1, random_state=SEED, stratify=train_pair["score"]
)

train_pair, test_pair = (
    Dataset.from_pandas(train_pair, preserve_index=False),
    Dataset.from_pandas(test_pair, preserve_index=False),
)

train_dataset = {"pair": train_pair, "triplet": train_triplet}

# ==== DEV DATASETS ====

dev_triplet_anc, dev_triplet_pos, dev_triplet_neg = (
    dev_triplet["anc_speech"].tolist(),
    dev_triplet["pos_speech"].tolist(),
    dev_triplet["neg_speech"].tolist(),
)

dev_triplet = pd.DataFrame(
    {
        "a_speech": dev_triplet_anc * 2,
        "b_speech": dev_triplet_pos + dev_triplet_neg,
        "score": [1.0] * len(dev_triplet_pos) + [0.0] * len(dev_triplet_neg),
    }
)

dev_dataset_kl = (
    pd.concat([dev_pair, dev_triplet])
    .drop_duplicates()
    .reset_index(drop=True)
    .sample(frac=1, random_state=SEED)
)
dev_dataset_spearman = pd.read_parquet(
    "hf://datasets/CATIE-AQ/frenchSTS/data/validation-00000-of-00001.parquet"
)


# ==== LOSSES ====

losses = {
    "pair": ContrastiveLoss(
        model, margin=MARGIN_LOSS_SIAMESE, distance_metric=DISTANCE_METRIC_SIAMESE
    ),
    "triplet": TripletLoss(
        model,
        triplet_margin=MARGIN_LOSS_TRIPLET,
        distance_metric=DISTANCE_METRIC_TRIPLET,
    ),
}

# ==== TRAINING ====

args = SentenceTransformerTrainingArguments(
    output_dir="checkpoints",
    num_train_epochs=4,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=2e-5,
    warmup_steps=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=500,
    logging_steps=100,
    use_cache=False,
    gradient_checkpointing=True,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
)

# ==== EVALUATORS ====

evaluator_spearman = EmbeddingSimilarityEvaluator(
    sentences1=dev_dataset_spearman["sentence1"].tolist(),
    sentences2=dev_dataset_spearman["sentence2"].tolist(),
    scores=dev_dataset_spearman["score"].tolist(),
    main_similarity=SimilarityFunction.COSINE,
    name="eval-spearman",
    show_progress_bar=True,
)


evaluator_kl = KLDivergenceEvaluator(
    sentences1=dev_dataset_kl["a_speech"].tolist(),
    sentences2=dev_dataset_kl["b_speech"].tolist(),
    scores=dev_dataset_kl["score"].tolist(),
    name="eval-kl",
    show_progress_bar=True,
)

trainer = SentenceTransformerTrainer(
    args=args,
    model=model,
    train_dataset=train_dataset,
    loss=losses,
    evaluator=SequentialEvaluator([evaluator_spearman, evaluator_kl]),
)

trainer.train()

model.save_pretrained("./adapter")
