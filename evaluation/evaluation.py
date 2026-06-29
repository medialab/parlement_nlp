import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import kl_div
from scipy.stats import pearsonr, spearmanr
from sentence_transformers.util import pairwise_cos_sim
from ast import literal_eval
from sklearn.metrics.pairwise import (
    cosine_distances,
    paired_cosine_distances,
    cosine_similarity,
)

KL_EPSILON = 1e-8

MAPPING_PARLEMENT_KL = {1.0: "AGREEMENT", 0.0: "DISAGREEMENT"}
MAPPING_PARLEMENT_SPEARMAN = {0.0: "DIFFERENT DEBATE", 1.0: "SAME DEBATE"}
MAPPING_SICK = {1: "NEUTRAL", 0: "ENTAILMENT", 2: "CONTRADICTION"}


def calculate_kl_div(a, b, scores, values=(0.0, 1.0)):
    similarities = pairwise_cos_sim(a, b).numpy()

    con, pro = values
    sim_0 = similarities[scores == con]
    sim_1 = similarities[scores == pro]
    bins = 100
    counts_0, _ = np.histogram(sim_0, bins=bins)
    counts_1, _ = np.histogram(sim_1, bins=bins)

    # Additive smoothing prevents divisions by zero and log(0) terms in KL.
    p = counts_0.astype(np.float64) + KL_EPSILON
    q = counts_1.astype(np.float64) + KL_EPSILON
    p /= np.sum(p)
    q /= np.sum(q)

    kl_all = kl_div(p, q)

    return np.nansum(kl_all)


def calcultate_spearman_pearson(a, b, scores):
    sims = pairwise_cos_sim(a, b).numpy()
    eval_spearman, _ = spearmanr(scores, sims)
    eval_pearson, _ = pearsonr(scores, sims)

    return eval_pearson, eval_spearman


def compute_stats_test(
    df, col_embedding_a, col_embedding_b, col_score, mapping=MAPPING_PARLEMENT_KL
):

    stats = {
        "all": {
            "ALL": np.array([]),
        }
    }

    for _, v in mapping.items():
        stats["all"][v] = np.array([])

    df["cosine_distance"] = 1 - paired_cosine_distances(
        df[col_embedding_a].tolist(), df[col_embedding_b].tolist()
    )

    stats["all"]["ALL"] = np.append(stats["all"]["ALL"], df["cosine_distance"])

    for k, v in mapping.items():
        stats["all"][v] = np.append(
            stats["all"][v], df[df[col_score] == k]["cosine_distance"]
        )

    return stats


def parlement(df: pd.DataFrame, name="qwen"):
    stats = compute_stats_test(
        df, "embedding_a_speech", "embedding_b_speech", "score", MAPPING_PARLEMENT_KL
    )

    labels = ["AGREEMENT", "DISAGREEMENT"]

    fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharey=True)

    for lbl in labels:
        vals = stats["all"].get(lbl, np.array([]))
        if vals.size:
            sns.kdeplot(vals, ax=axes, label=lbl, fill=True, alpha=0.3)

    axes.set_xlabel("Cosine similarity")
    axes.set_ylabel("Density")

    handles, legend_labels = axes.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", ncol=3)

    plt.suptitle("Cosine similarity distributions")

    plt.savefig(f"./figures/parlement_{name}.jpg")

    embeddings_a = np.array(df["embedding_a_speech"].tolist())
    embeddings_b = np.array(df["embedding_b_speech"].tolist())
    scores = np.array(df["score"])

    kl = calculate_kl_div(embeddings_a, embeddings_b, scores)

    print("==== PARLEMENT TEST SET - KL DIV ====")
    print("KL-Divergence :", kl)
    print(f"Plot of distances distribution saved to ./figures/parlement_{name}.jpg")
    print("")

    return kl


def catie_sts(df: pd.DataFrame):

    print("==== Catie-AQ/STS ====")

    metrics = {}

    for dataset, group in df.groupby("dataset"):
        embeddings_a = np.array(group["embedding_sentence1"].tolist())
        embeddings_b = np.array(group["embedding_sentence2"].tolist())
        scores = np.array(group["score"])

        pearson, spearman = calcultate_spearman_pearson(
            embeddings_a, embeddings_b, scores
        )

        print(
            f"[{dataset}] pearson correlation :",
            pearson,
            "; spearman correlation :",
            spearman,
        )

        metrics[dataset] = (pearson, spearman)

    print("")

    return metrics


def spearman(df: pd.DataFrame, name="qwen"):
    print("==== PARLEMENT TEST SET - SPEARMAN ====")

    stats = compute_stats_test(
        df,
        "embedding_a_speech",
        "embedding_b_speech",
        "score",
        MAPPING_PARLEMENT_SPEARMAN,
    )

    labels = ["DIFFERENT DEBATE", "SAME DEBATE"]

    fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharey=True)

    for lbl in labels:
        vals = stats["all"].get(lbl, np.array([]))
        if vals.size:
            sns.kdeplot(vals, ax=axes, label=lbl, fill=True, alpha=0.3)

    axes.set_xlabel("Cosine similarity")
    axes.set_ylabel("Density")

    handles, legend_labels = axes.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", ncol=3)

    plt.suptitle("Cosine similarity distributions")

    plt.savefig(f"./figures/parlement_2_{name}.jpg")

    embeddings_a = np.array(df["embedding_a_speech"].tolist())
    embeddings_b = np.array(df["embedding_b_speech"].tolist())
    scores = np.array(df["score"])

    pearson, spearman = calcultate_spearman_pearson(embeddings_a, embeddings_b, scores)

    print("Pearson correlation :", pearson, "; spearman correlation :", spearman)
    print("")

    return (pearson, spearman)


def sick(df: pd.DataFrame, name="qwen"):
    stats = compute_stats_test(
        df, "embedding_sentence_A", "embedding_sentence_B", "label", MAPPING_SICK
    )

    labels = ["NEUTRAL", "ENTAILMENT", "CONTRADICTION"]

    fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharey=True)

    for lbl in labels:
        vals = stats["all"].get(lbl, np.array([]))
        if vals.size:
            sns.kdeplot(vals, ax=axes, label=lbl, fill=True, alpha=0.3)

    axes.set_xlabel("Cosine similarity")
    axes.set_ylabel("Density")

    handles, legend_labels = axes.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", ncol=3)

    plt.suptitle("Cosine similarity distributions")

    plt.savefig(f"./figures/sick_{name}.jpg")

    # embeddings_a = np.array(df["embedding_sentence_A"].tolist())
    # embeddings_b = np.array(df["embedding_sentence_B"].tolist())
    # scores = np.array(df["label"])

    # kl = calculate_kl_div(embeddings_a, embeddings_b, scores)

    print("==== SICK TEST SET ====")
    # print("KL-Divergence :", kl)
    print(f"Plot of distances distribution saved to ./figures/sick_{name}.jpg")
    print("")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parlement",
        help="CSV path for parlement embeddings",
        default=None,
    )
    parser.add_argument(
        "--sts",
        help="CSV path for Catie-AQ/STS embeddings",
        default=None,
    )
    parser.add_argument(
        "--sick",
        help="CSV path for SICK-Fr embeddings",
        default=None,
    )
    parser.add_argument(
        "--spearman",
        help="CSV path for parlement spearman embeddings",
        default=None,
    )
    parser.add_argument(
        "--model",
        help="Model name to evaluate",
        default="qwen",
    )
    args = parser.parse_args()

    if args.parlement:
        df = pd.read_csv(
            args.parlement,
            converters={
                "embedding_a_speech": literal_eval,
                "embedding_b_speech": literal_eval,
            },
        )
        results = parlement(df, name=args.model)
        # TODO

    if args.sts:
        df = pd.read_csv(
            args.sts,
            converters={
                "embedding_sentence1": literal_eval,
                "embedding_sentence2": literal_eval,
            },
        )
        results = catie_sts(df)
        # TODO

    if args.sick:
        df = pd.read_csv(
            args.sick,
            converters={
                "embedding_sentence_A": literal_eval,
                "embedding_sentence_B": literal_eval,
            },
        )
        results = sick(df, name=args.model)
        # TODO

    if args.spearman:
        df = pd.read_csv(
            args.spearman,
            converters={
                "embedding_a_speech": literal_eval,
                "embedding_b_speech": literal_eval,
            },
        )
        results = spearman(df, name=args.model)
