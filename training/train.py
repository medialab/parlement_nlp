import logging
import csv
import os
import numpy as np
import shutil
import sys

import optuna
import torch
from transformers import EarlyStoppingCallback, TrainerCallback

from scipy.special import kl_div

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


logger = logging.getLogger(__name__)


KL_EPSILON = 1e-8
KL_HARMONIC_SCALE = 1.0


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

        err(
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

        # Additive smoothing prevents divisions by zero and log(0) terms in KL.
        p = counts_0.astype(np.float64) + KL_EPSILON
        q = counts_1.astype(np.float64) + KL_EPSILON
        p /= np.sum(p)
        q /= np.sum(q)

        kl_all = kl_div(p, q)
        kl = np.nansum(kl_all)

        metrics = {"kl_div_cosine": kl}
        err(f"COSINE-KL-Divergence: {kl:.4f}")

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

                writer.writerow([epoch, steps, metrics["kl_div_cosine"]])

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


QUICK = False
RESUME = True

SEED = 24
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda"
CHECKPOINT_ROOT_DIR = "checkpoints"
OPTUNA_TRIALS = 20
OPTUNA_STUDY_NAME = "sentence_transformer_optuna_multiobjective"
OPTUNA_STORAGE = "sqlite:///optuna_study.db"
OBJECTIVE_METRIC_SPEARMAN = "eval-spearman_spearman_cosine"
OBJECTIVE_METRIC_KL = "eval-kl_kl_div_cosine"
OBJECTIVE_METRIC_HARMONIC = "eval_harmonic_mean"
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_THRESHOLD = 1e-2
EVAL_SAVE_STEPS = 500 if not QUICK else 250

# Hyperparameter sets initialization for Optuna.
HPARAM_BATCH_SIZE = 1
HPARAM_GRADIENT_ACCUMULATION_STEPS = 8
HPARAM_LEARNING_RATE = 3e-5
HPARAM_NUM_EPOCHS = 4
HPARAM_MARGIN_LOSS_SIAMESE = [0.2, 0.4, 0.6]
HPARAM_MARGIN_LOSS_TRIPLET = [0.2, 0.4, 0.6]
HPAREM_DATA_FILTER_PAIR = [0.0, 0.25, 0.33]
HPAREM_DATA_FILTER_TRIPLET = [0.0, 0.25, 0.33]
HPARAM_LORA_R = 4
# HPARAM_LORA_ALPHA = [8, 16]
HPARAM_LORA_DROPOUT = 0.05

DISTANCE_METRIC_SIAMESE = SiameseDistanceMetric.COSINE_DISTANCE
DISTANCE_METRIC_TRIPLET = TripletDistanceMetric.COSINE


modules = {
    "Qwen/Qwen3-Embedding-0.6B": ["q_proj", "k_proj", "v_proj"],
    "Lajavaness/sentence-camembert-large": ["query", "key", "value"],
}

# ==== DATASETS ====

train_pair_df = pd.read_csv("./splits/train_paires.csv")
train_triplet_df = pd.read_csv("./splits/train_triplets.csv")
dev_dataset_kl = pd.read_csv("./splits/dev_kl.csv")
dev_dataset_spearman = pd.read_csv("./splits/dev_spearman.csv")

pair_quick_nb = int(len(train_pair_df) * 0.1)
triplet_quick_nb = int(len(train_triplet_df) * 0.1)

def err(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def _extract_multi_objective_metrics(metrics):
    if OBJECTIVE_METRIC_SPEARMAN in metrics and OBJECTIVE_METRIC_KL in metrics:
        return float(metrics[OBJECTIVE_METRIC_SPEARMAN]), float(
            metrics[OBJECTIVE_METRIC_KL]
        )

    spearman_value = None
    kl_value = None

    for key, value in metrics.items():
        key_lower = key.lower()
        if spearman_value is None and "spearman" in key_lower:
            spearman_value = float(value)
        if kl_value is None and "kl" in key_lower and "div" in key_lower:
            kl_value = float(value)

    if spearman_value is None or kl_value is None:
        raise ValueError(
            "Could not extract both Spearman and KL-divergence metrics from evaluation output: "
            f"{metrics}"
        )

    return spearman_value, kl_value


def _compute_harmonic_objective(spearman: float, kl_divergence: float) -> float:
    """Compute a stable trade-off score used for checkpoint selection.

    Spearman is already bounded in [-1, 1], while KL is unbounded and can drift
    over training. We map KL to [0, 1) with kl/(kl+scale) so harmonic mean stays
    stable across eval steps and does not depend on previous maxima.
    """
    if spearman <= 0.0 or kl_divergence <= 0.0:
        return 0.0

    kl_score = kl_divergence / (kl_divergence + KL_HARMONIC_SCALE)
    if kl_score <= 0.0:
        return 0.0

    return (2.0 * spearman * kl_score) / (spearman + kl_score)


class DoubleObjectiveCallBack(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        state._best_spearman = -99
        state._best_kl = 0.0
        state._best_harmonic = -99
        state._best_step = -99
        return control

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        spearman, kl = _extract_multi_objective_metrics(metrics)

        harmonic_mean = _compute_harmonic_objective(spearman, kl)

        err("Harmonic mean : %.2f" % harmonic_mean)

        metrics[OBJECTIVE_METRIC_HARMONIC] = harmonic_mean

        state._best_spearman = max(state._best_spearman, spearman)
        state._best_kl = max(state._best_kl, kl)
        if harmonic_mean > state._best_harmonic:
            state._best_harmonic = harmonic_mean
            state._best_step = state.global_step

        return control
    
    


def build_model(lora_r: int, lora_alpha: int, lora_dropout: float):
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        target_modules=modules[MODEL_NAME],
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model.add_adapter(peft_config)
    return model


def build_trainer(trial: optuna.Trial):
    #per_device_train_batch_size = trial.suggest_categorical(
    #    "per_device_train_batch_size", HPARAM_BATCH_SIZE
    #)
    #gradient_accumulation_steps = trial.suggest_categorical(
    #    "gradient_accumulation_steps", HPARAM_GRADIENT_ACCUMULATION_STEPS
    #)
    #learning_rate = trial.suggest_categorical("learning_rate", HPARAM_LEARNING_RATE)
    #num_train_epochs = trial.suggest_categorical("num_train_epochs", HPARAM_NUM_EPOCHS)
    margin_loss_siamese = trial.suggest_categorical(
        "margin_loss_siamese", HPARAM_MARGIN_LOSS_SIAMESE
    )
    margin_loss_triplet = trial.suggest_categorical(
        "margin_loss_triplet", HPARAM_MARGIN_LOSS_TRIPLET
    )
    #lora_r = trial.suggest_categorical("lora_r", HPARAM_LORA_R)
    #lora_dropout = trial.suggest_categorical("lora_dropout", HPARAM_LORA_DROPOUT)

    data_filter_pair = trial.suggest_categorical(
        "data_filter_pair", HPAREM_DATA_FILTER_PAIR
    )
    data_filter_triplet = trial.suggest_categorical(
        "data_filter_triplet", HPAREM_DATA_FILTER_TRIPLET
    )

    filtered_train_pair_df = train_pair_df[
        train_pair_df["cosine"] >= data_filter_pair
    ]

    filtered_train_triplet_df = train_triplet_df[
        train_triplet_df["cosine"] >= data_filter_triplet
    ]

    filtered_train_pair_df = filtered_train_pair_df[["a_speech", "b_speech", "score"]]
    filtered_train_triplet_df = filtered_train_triplet_df[
        ["anc_speech", "pos_speech", "neg_speech"]
    ]

    if QUICK:
        filtered_train_pair_df = filtered_train_pair_df.sample(
            n=min(len(filtered_train_pair_df), pair_quick_nb), random_state=SEED
        )
        filtered_train_triplet_df = filtered_train_triplet_df.sample(
            n=min(len(filtered_train_triplet_df), triplet_quick_nb), random_state=SEED
        )


    model = build_model(
        lora_r=HPARAM_LORA_R, lora_alpha=int(HPARAM_LORA_R) * 2, lora_dropout=HPARAM_LORA_DROPOUT
    )

    train_dataset = {
        "pair": Dataset.from_pandas(filtered_train_pair_df, preserve_index=False),
        "triplet": Dataset.from_pandas(filtered_train_triplet_df, preserve_index=False),
    }

    losses = {
        "pair": ContrastiveLoss(
            model,
            margin=margin_loss_siamese,
            distance_metric=DISTANCE_METRIC_SIAMESE,
        ),
        "triplet": TripletLoss(
            model,
            triplet_margin=margin_loss_triplet,
            distance_metric=DISTANCE_METRIC_TRIPLET,
        ),
    }

    total = len(filtered_train_pair_df) + len(filtered_train_triplet_df)
    err("==== NEW TRIAL ====")
    err("Number of items :", total)
    err("Number of steps", (total // HPARAM_GRADIENT_ACCUMULATION_STEPS) * HPARAM_NUM_EPOCHS)

    args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(CHECKPOINT_ROOT_DIR, f"trial-{trial.number}"),
        num_train_epochs=HPARAM_NUM_EPOCHS,
        per_device_train_batch_size=HPARAM_BATCH_SIZE,
        per_device_eval_batch_size=HPARAM_BATCH_SIZE,
        learning_rate=HPARAM_LEARNING_RATE,
        warmup_steps=0.1,
        fp16=False,
        bf16=False,
        eval_strategy="steps",
        eval_steps=EVAL_SAVE_STEPS,
        save_strategy="steps",
        save_steps=EVAL_SAVE_STEPS,
        logging_steps=100,
        metric_for_best_model=OBJECTIVE_METRIC_HARMONIC,
        greater_is_better=True,
        load_best_model_at_end=True,
        use_cache=False,
        gradient_checkpointing=True,
        gradient_accumulation_steps=HPARAM_GRADIENT_ACCUMULATION_STEPS,
    )

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

    return SentenceTransformerTrainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        loss=losses,
        evaluator=SequentialEvaluator([evaluator_spearman, evaluator_kl]),
        callbacks=[
            DoubleObjectiveCallBack(),
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            ),
        ],
    )


def objective(trial):
    trial_output_dir = os.path.join(CHECKPOINT_ROOT_DIR, f"trial-{trial.number}")
    if os.path.isdir(trial_output_dir):
        shutil.rmtree(trial_output_dir)

    trainer = build_trainer(trial)
    trainer.train()
    metrics = trainer.evaluate()
    spearman_value, kl_value = _extract_multi_objective_metrics(metrics)
    trial.set_user_attr("metrics", metrics)
    trial.set_user_attr("objective_spearman", spearman_value)
    trial.set_user_attr("objective_kl", kl_value)

    trainer.model.save_pretrained(os.path.join(trial_output_dir, "adapter"))

    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return spearman_value, kl_value


def run_optuna_experiment():
    os.makedirs(CHECKPOINT_ROOT_DIR, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        directions=["maximize", "maximize"],
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS)

    print("pareto_trials_count=", len(study.best_trials))
    for trial in study.best_trials:
        print(
            "pareto_trial=",
            trial.number,
            "values=",
            trial.values,
            "params=",
            trial.params,
        )

    return study


if __name__ == "__main__":
    run_optuna_experiment()
