import argparse
import casanova
import pandas as pd
import sys
import numpy as np
import torch
import random
import traceback
import os
from collections import deque

from transformers import TrainerCallback, set_seed

from scipy.special import kl_div
from scipy.stats import pearsonr, spearmanr

from datetime import datetime

from os.path import join
from os import makedirs

from accelerate import prepare_pippy

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.util import pairwise_cos_sim
from sentence_transformers.sentence_transformer.losses import (
    SiameseDistanceMetric,
    TripletDistanceMetric
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

# Cf. https://github.com/yqhu/profiler-workshop/blob/c8d4a7c30a61cc7b909d89f88f5fd36b70c55769/hf_training_trainer_prof.py
class ProfCallback(TrainerCallback):
    def __init__(self):
        #self.prof = prof
        pass

    def on_step_end(self, args, state, control, **kwargs):
        #print("STEP END")
        #self.prof.step()
        torch.cuda.memory._dump_snapshot(f"trace/trace_{state.global_step}.pkl")

class KLDivergenceEvaluator(BaseEvaluator):
    def __init__(
        self,
        sentences1,
        sentences2,
        scores,
        batch_size=16,
        name="",
        show_progress_bar=False,
        write_csv=True,
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

        # === KL DIV ===

        sim_0 = similarities[scores == 0.5]
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

        # === PEARSON / SPEARMAN

        eval_pearson, _ = pearsonr(similarities, scores)
        eval_spearman, _ = spearmanr(similarities, scores)

        metrics["pearson_cosine"] = eval_pearson
        metrics["spearman_cosine"] = eval_spearman

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
        sts_spearman = None
        sts_pearson = None
        epoch = None

        for key, value in metrics.items():
            if "kl_div" in key:
                kl = value
            if "pearson_cosine" and "sts" in key:
                sts_pearson = value
            if "spearman_cosine" and "sts" in key:
                sts_spearman = value
            if "pearson_cosine" and "val" in key:
                parl_pearson = value
            if "spearman_cosine" and "val" in key:
                parl_spearman = value
            if "epoch" in key:
                epoch = value

        dt = datetime.now().replace(microsecond=0).isoformat()

        self.log_fn(
            (
                dt,
                epoch,
                self.last_step,
                self.last_loss,
                self.last_lr,
                kl,
                sts_spearman,
                sts_pearson,
                parl_spearman,
                parl_pearson,
            )
        )

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


class MemoryCoSENTLoss(torch.nn.Module):
    def __init__(self, model, scale=20.0, similarity_fct=pairwise_cos_sim, memory_size=64):
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.memory_size = memory_size
        self.memory_embeddings_a = deque(maxlen=memory_size)
        self.memory_embeddings_b = deque(maxlen=memory_size)
        self.memory_labels = deque(maxlen=memory_size)

    def forward(self, sentence_features, labels):
        current_a = self.model(sentence_features[0])["sentence_embedding"]
        current_b = self.model(sentence_features[1])["sentence_embedding"]
        current_labels = labels.view(-1)

        if len(self.memory_labels) == 0:
            self._save_in_memory(current_a, current_b, current_labels)
            # Keep a graph-connected zero on the very first step.
            return (current_a.sum() + current_b.sum()) * 0.0

        memory_a = torch.cat(list(self.memory_embeddings_a), dim=0).to(current_a.device)
        memory_b = torch.cat(list(self.memory_embeddings_b), dim=0).to(current_b.device)
        memory_labels = torch.cat(list(self.memory_labels), dim=0).to(current_labels.device)

        all_a = torch.cat([memory_a, current_a], dim=0)
        all_b = torch.cat([memory_b, current_b], dim=0)
        all_labels = torch.cat([memory_labels, current_labels], dim=0)

        scores = self.similarity_fct(all_a, all_b)
        scores = scores * self.scale
        scores = scores[:, None] - scores[None, :]

        label_matrix = all_labels[:, None] < all_labels[None, :]
        label_matrix = label_matrix.float()

        scores = scores - (1 - label_matrix) * 1e12

        scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
        loss = torch.logsumexp(scores, dim=0)

        self._save_in_memory(current_a, current_b, current_labels)

        return loss

    def _save_in_memory(self, embeddings_a, embeddings_b, labels):
        self.memory_embeddings_a.append(embeddings_a.detach().cpu())
        self.memory_embeddings_b.append(embeddings_b.detach().cpu())
        self.memory_labels.append(labels.detach().cpu())




MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODULES = {
    "Qwen/Qwen3-Embedding-0.6B": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "Lajavaness/sentence-camembert-large": ["query", "key", "value"],
}
DEVICE = "cuda"
DISTANCE_METRIC_SIAMESE = SiameseDistanceMetric.COSINE_DISTANCE
DISTANCE_METRIC_TRIPLET = TripletDistanceMetric.COSINE
BATCH_SIZE = 32
NUM_EPOCHS = 2
EVAL_SAVE_STEPS = 500
GRADIENT_ACCUMULATION_STEPS = 1

TRAINING_ERROR_CSV_HEADER = ["trial", "message"]
HPARAM_CSV_INPUT = "hyperparams.csv"
HPARAM_CSV_HEADER = [
    "datetime",
    "epoch",
    "step",
    "loss",
    "dynamic_lr",
    "kl_divergence",
    "sts_spearman_cosine",
    "sts_pearson_cosine",
    "parlement_spearman_cosine",
    "parlement_pearson_cosine",
]


def err(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
    print(output)
    os.makedirs("trace", exist_ok=True)
    p.export_chrome_trace("./trace/trace_" + str(p.step_num) + ".json")

def trial(
    params,
    datasets,
    log_callback,
    model=MODEL_NAME,
    batch_size=BATCH_SIZE,
    eval_steps=EVAL_SAVE_STEPS,
    checkpoints_dir=False,
    accumumation_steps=GRADIENT_ACCUMULATION_STEPS,
):
    
  
    # === datasets ===
    train_df, dev_df, dev_spearman = datasets

    # === hyper params ===
    i, lr, scale_loss, df_pair, df_triplet, lora_r = params
    i, lr, scale_loss, df_pair, df_triplet, lora_r = (
        int(i),
        float(lr),
        float(scale_loss),
        float(df_pair),
        float(df_triplet),
        int(lora_r),
    )

    # === filtering of datasets ===
    filtered_train_df = train_df[train_df["cosine"] >= df_pair]
    filtered_train_df = filtered_train_df[["a_speech", "b_speech", "score"]]
    

    # === model instanciation ===
    model = SentenceTransformer(model, device=DEVICE)
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        target_modules=MODULES.get(model, MODULES.get(MODEL_NAME)),
        r=lora_r,
        lora_alpha=lora_r,
        lora_dropout=0.05,
    )
    model.add_adapter(peft_config)

    model = prepare_pippy(model)

    train = Dataset.from_pandas(filtered_train_df, preserve_index=False)
    loss = MemoryCoSENTLoss(model, scale=scale_loss, memory_size=accumumation_steps)

    total = len(filtered_train_df)

    err("==== NEW TRIAL ====")
    err("Number of items :", total)
    err("Number of steps", ((total // batch_size) // accumumation_steps) * NUM_EPOCHS)

    args = SentenceTransformerTrainingArguments(
        output_dir=join(checkpoints_dir, f"trial-{i}") if checkpoints_dir else None,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=0.1,
        fp16=False,
        bf16=False,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps" if checkpoints_dir else "no",
        save_steps=eval_steps,
        logging_steps=100,
        eval_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=accumumation_steps,
        dataloader_drop_last=True,
        seed=SEED,
        use_cache=False
    )

    print(model.device, args.place_model_on_device, args.device)

    evaluator_sts = EmbeddingSimilarityEvaluator(
        sentences1=dev_spearman["sentence1"].tolist(),
        sentences2=dev_spearman["sentence2"].tolist(),
        scores=dev_spearman["score"].tolist(),
        main_similarity=SimilarityFunction.COSINE,
        name="sts",
        show_progress_bar=True,
        batch_size=1,
    )

    evaluator_kl = KLDivergenceEvaluator(
        sentences1=dev_df["a_speech"].tolist(),
        sentences2=dev_df["b_speech"].tolist(),
        scores=dev_df["score"].tolist(),
        name="val",
        show_progress_bar=True,
        batch_size=1,
    )

    """
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1),
        on_trace_ready=trace_handler,
    ) as prof:
    """

    trainer = SentenceTransformerTrainer(
        args=args,
        model=model,
        train_dataset=train,
        loss=loss,
        evaluator=SequentialEvaluator(
            [evaluator_sts, evaluator_kl]
        ),
        callbacks=[
            LoggingCallBack(log_callback),
        ],
    )

    trainer.train()

    del trainer
    del model
    del loss

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hyperparams",
        default=HPARAM_CSV_INPUT,
        help="Path to hyperparameters CSV file.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Name or path of the SBERT model (default to Qwen/Qwen3-Embedding-0.6B)"
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
    parser.add_argument(
        "-c",
        "--checkpoints",
        action="store_true",
        help="Save checkpoints along training.",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
        help="Gradient accumulation steps.",
    )
    cli_args = parser.parse_args()

    iso_dt = (
        datetime.now()
        .replace(microsecond=0)
        .isoformat()
        .replace("T", "_")
        .replace(":", "-")
    )
    csv_output_path = cli_args.hyperparams.replace(".csv", "") + f"_{iso_dt}.csv"
    csv_output_error_path = (
        cli_args.hyperparams.replace(".csv", "") + f"_{iso_dt}_error.csv"
    )

    makedirs("logs", exist_ok=True)

    csv_output_path = join("./logs", csv_output_path)
    csv_output_error_path = join("./logs", csv_output_error_path)

    if cli_args.checkpoints:
        checkpoints_dir = join("checkpoints", iso_dt)
        makedirs(checkpoints_dir, exist_ok=True)
    else:
        checkpoints_dir = None

    err("Loading datasets files...")

    train_df = pd.read_csv("./splits/train_cosent.csv")
    dev_df = pd.read_csv("./splits/dev_cosent.csv")
    dev_sts_df = pd.read_csv("./splits/dev_spearman.csv")

    with (
        casanova.enricher(
            cli_args.hyperparams, csv_output_path, add=HPARAM_CSV_HEADER
        ) as enricher,
        casanova.writer(csv_output_error_path, TRAINING_ERROR_CSV_HEADER) as error,
    ):
        for row in enricher:
            trial_i = int(row[0])

            if trial_i < cli_args.start_trial_index:
                continue

            def callback(logs):
                enricher.writerow(row, add=logs)

            datasets = (
                train_df,
                dev_df,
                dev_sts_df,
            )

            try:
                trial(
                    row,
                    datasets,
                    callback,
                    model=cli_args.model,
                    batch_size=cli_args.batch_size,
                    eval_steps=cli_args.eval_steps,
                    checkpoints_dir=checkpoints_dir,
                    accumumation_steps=cli_args.accumulation_steps,
                )
            except Exception as e:
                err("===== ERROR - STOPPING TRIAL =====")
                err(str(e))
                error.writerow([trial_i, traceback.format_exc()])
