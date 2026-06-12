import argparse
import casanova
import pandas as pd
import sys
import numpy as np
import torch
import random

from transformers import TrainerCallback, set_seed

from scipy.special import kl_div

from datetime import datetime

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
from datasets import Dataset
from peft import LoraConfig, TaskType

KL_EPSILON = 1e-8

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
set_seed(SEED)

class KLDivergenceEvaluator(BaseEvaluator):
    def __init__(
        self,
        sentences1,
        sentences2,
        scores,
        batch_size = 16,
        name = "",
        show_progress_bar = False,
        write_csv = True,
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
        self.show_progress_bar = show_progress_bar

        self.csv_headers = ["epoch", "steps", "cosine_kl_div"]

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        embeddings1 = self.embed_inputs(model, self.sentences1)
        embeddings2 = self.embed_inputs(model, self.sentences2)

        similarities = pairwise_cos_sim(embeddings1, embeddings2).detach().cpu().numpy()
        scores = np.array(self.scores)

        sim_0 = similarities[scores == 0.0]
        sim_1 = similarities[scores == 1.0]
        bins = 100
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

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)

        return metrics

    def embed_inputs(self, model, sentences, **kwargs):
        return model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            **kwargs,
        )

    @property
    def description(self):
        return "KL Divergence"

class LoggingCallBack(TrainerCallback):
    def __init__(self, log_fn):
        self.log_fn = log_fn

        self.last_step = 0
        self.last_loss = 0.0
        self.last_lr = 0.0
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        kl = None
        spearman = None
        pearson = None
        epoch = None

        for key, value in metrics.items():
            if "kl_div" in key:
                kl = value
            if "pearson_cosine" in key:
                pearson = value
            if "spearman_cosine" in key:
                spearman = value
            if "epoch" in key:
                epoch = value

        dt = datetime.now().replace(microsecond=0).isoformat()
        
        self.log_fn((dt, epoch, self.last_step, self.last_loss, self.last_lr, kl, spearman, pearson))

        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        self.last_step = state.global_step
        return control
    
    def on_log(self, args, state, control, logs, **kwargs):
        for key, value in logs.items():
            if "loss" in key:
                self.last_loss = value
            if "learning_rate" in key:
                self.last_lr = value
        return control


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODULES = {
    "Qwen/Qwen3-Embedding-0.6B": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    "Lajavaness/sentence-camembert-large": ["query", "key", "value"],
}
DEVICE = "cuda"
DISTANCE_METRIC_SIAMESE = SiameseDistanceMetric.COSINE_DISTANCE
DISTANCE_METRIC_TRIPLET = TripletDistanceMetric.COSINE
BATCH_SIZE = 32
NUM_EPOCHS = 3
EVAL_SAVE_STEPS = 500

TRAINING_ERROR_CSV_HEADER = [
    "trial",
    "message"
]
HPARAM_CSV_INPUT = "hyperparams.csv"
HPARAM_CSV_HEADER = [
    "datetime",
    "epoch",
    "step",
    "loss",
    "dynamic_lr",
    "kl_divergence",
    "spearman_cosine",
    "pearson_cosine"
]

def err(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def trial(params, datasets, log_callback, batch_size=BATCH_SIZE, eval_steps=EVAL_SAVE_STEPS):
    # === datasets ===
    train_pair_df, train_triplet_df, dev_dataset_kl, dev_dataset_spearman = datasets

    # === hyper params ===
    i, lr, m_loss_siamese, m_loss_triplet, df_pair, df_triplet, lora_r = params
    i, lr, m_loss_siamese, m_loss_triplet, df_pair, df_triplet, lora_r = (
        int(i),
        float(lr),
        float(m_loss_siamese),
        float(m_loss_triplet),
        float(df_pair),
        float(df_triplet),
        int(lora_r)
    )

    # === filtering of datasets ===
    filtered_train_pair_df = train_pair_df[
        train_pair_df["cosine"] >= df_pair
    ]

    filtered_train_triplet_df = train_triplet_df[
        train_triplet_df["cosine"] >= df_triplet
    ]

    filtered_train_pair_df = filtered_train_pair_df[
        ["a_speech", "b_speech", "score"]
    ]
    filtered_train_triplet_df = filtered_train_triplet_df[
        ["anc_speech", "pos_speech", "neg_speech"]
    ]

    # === model instanciation === 
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        target_modules=MODULES[MODEL_NAME],
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
    )
    model.add_adapter(peft_config)

    train_dataset = {
        "pair": Dataset.from_pandas(filtered_train_pair_df, preserve_index=False),
        "triplet": Dataset.from_pandas(filtered_train_triplet_df, preserve_index=False),
    }

    losses = {
        "pair": ContrastiveLoss(
            model,
            margin=m_loss_siamese,
            distance_metric=DISTANCE_METRIC_SIAMESE,
        ),
        "triplet": TripletLoss(
            model,
            triplet_margin=m_loss_triplet,
            distance_metric=DISTANCE_METRIC_TRIPLET,
        ),
    }

    total = len(filtered_train_pair_df) + len(filtered_train_triplet_df)

    err("==== NEW TRIAL ====")
    err("Number of items :", total)
    err("Number of steps", (total // batch_size) * NUM_EPOCHS)

    args = SentenceTransformerTrainingArguments(
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=0.1,
        fp16=False,
        bf16=False,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="no",
        logging_steps=100,
        gradient_checkpointing=False,
        seed=SEED
    )

    evaluator_spearman = EmbeddingSimilarityEvaluator(
        sentences1=dev_dataset_spearman["sentence1"].tolist(),
        sentences2=dev_dataset_spearman["sentence2"].tolist(),
        scores=dev_dataset_spearman["score"].tolist(),
        main_similarity=SimilarityFunction.COSINE,
        name="sts",
        show_progress_bar=True,
    )

    evaluator_kl = KLDivergenceEvaluator(
        sentences1=dev_dataset_kl["a_speech"].tolist(),
        sentences2=dev_dataset_kl["b_speech"].tolist(),
        scores=dev_dataset_kl["score"].tolist(),
        name="val",
        show_progress_bar=True,
    )

    trainer = SentenceTransformerTrainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        loss=losses,
        evaluator=SequentialEvaluator([evaluator_spearman, evaluator_kl]),
        callbacks=[
            LoggingCallBack(log_callback),
        ],
    )

    trainer.train()

    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hyperparams",
        default=HPARAM_CSV_INPUT,
        help="Path to hyperparameters CSV file.",
    )
    parser.add_argument(
        "--start-trial-index",
        type=int,
        default=0,
        help="Trial index to start from.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size of training.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=EVAL_SAVE_STEPS,
        help="Eval save steps.",
    )
    cli_args = parser.parse_args()

    iso_dt = datetime.now().replace(microsecond=0).isoformat().replace('T', '_').replace(':', '-')
    csv_output_path = cli_args.hyperparams.replace(".csv", "") + f"_{iso_dt}.csv"
    csv_output_error_path = cli_args.hyperparams.replace(".csv", "") + f"_{iso_dt}_error.csv"

    err("Loading datasets files...")

    train_pair_df = pd.read_csv("./splits/train_paires.csv")
    train_triplet_df = pd.read_csv("./splits/train_triplets.csv")
    dev_dataset_kl = pd.read_csv("./splits/dev_kl.csv")
    dev_dataset_spearman = pd.read_csv("./splits/dev_spearman.csv")

    with (
        casanova.enricher(cli_args.hyperparams, csv_output_path, add=HPARAM_CSV_HEADER) as enricher, 
        casanova.writer(csv_output_error_path, TRAINING_ERROR_CSV_HEADER) as error
    ):
        for row in enricher:
            trial_i = int(row[0])

            if trial_i < cli_args.start_trial_index:
                continue
            
            def callback(logs):
                enricher.writerow(row, add=logs)

            datasets = (
                train_pair_df,
                train_triplet_df,
                dev_dataset_kl,
                dev_dataset_spearman
            )

            try:
                trial(row, datasets, callback, batch_size=cli_args.batch_size, eval_steps=cli_args.eval_steps)
            except Exception as e:
                err("===== ERROR - STOPPING TRIAL =====")
                err(str(e))
                error.writerow([
                    trial_i,
                    str(e)
                ])


