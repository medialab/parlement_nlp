from sklearn.model_selection import train_test_split

from sentence_transformers.util import pairwise_cos_sim
from sentence_transformers import (
    SentenceTransformer,
)

import torch
import numpy as np

import pandas as pd

SEED = 24


dataset_triplet = pd.read_csv(
    "./data-4096/triplets.csv", dtype={"agreement": "float64"}
)
dataset_triplet = dataset_triplet[["anc_speech", "pos_speech", "neg_speech"]]

dataset_pair = pd.read_csv("./data-4096/paires.csv", dtype={"agreement": "float64"})
dataset_pair = dataset_pair[["a_speech", "b_speech", "agreement"]]
dataset_pair = dataset_pair.rename(columns={"agreement": "score"})

# splitting

train_triplet, test_triplet = train_test_split(
    dataset_triplet, test_size=0.1, random_state=SEED
)
train_triplet, dev_triplet = train_test_split(
    train_triplet, test_size=0.1, random_state=SEED
)

train_pair, test_pair = train_test_split(
    dataset_pair, test_size=0.1, random_state=SEED, stratify=dataset_pair["score"]
)
train_pair, dev_pair = train_test_split(
    train_pair, test_size=0.1, random_state=SEED, stratify=train_pair["score"]
)


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

# ==== TEST DATASETS ====

test_triplet_anc, test_triplet_pos, test_triplet_neg = (
    test_triplet["anc_speech"].tolist(),
    test_triplet["pos_speech"].tolist(),
    test_triplet["neg_speech"].tolist(),
)

test_triplet = pd.DataFrame(
    {
        "a_speech": test_triplet_anc * 2,
        "b_speech": test_triplet_pos + test_triplet_neg,
        "score": [1.0] * len(test_triplet_pos) + [0.0] * len(test_triplet_neg),
    }
)

test_dataset_kl = (
    pd.concat([test_pair, test_triplet])
    .drop_duplicates()
    .reset_index(drop=True)
    .sample(frac=1, random_state=SEED)
)
test_dataset_spearman = pd.read_parquet(
    "hf://datasets/CATIE-AQ/frenchSTS/data/test-00000-of-00001.parquet"
)

# ==== COSINE SIM ====

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

train_texts = list(set(
    train_pair["a_speech"].tolist()
    + train_pair["b_speech"].tolist()
    + train_triplet["anc_speech"].tolist()
    + train_triplet["pos_speech"].tolist()
    + train_triplet["neg_speech"].tolist()
))

train_embeddings_array = model.encode(train_texts, batch_size=16, show_progress_bar=True)
train_embeddings = {text: embedding for text, embedding in zip(train_texts, train_embeddings_array)}

embeddings_pair_a = torch.from_numpy(
    np.stack(train_pair["a_speech"].map(train_embeddings).to_numpy()).astype(np.float32)
)
embeddings_pair_b = torch.from_numpy(
    np.stack(train_pair["b_speech"].map(train_embeddings).to_numpy()).astype(np.float32)
)
pair_sim = pairwise_cos_sim(embeddings_pair_a, embeddings_pair_b)
if pair_sim.ndim == 2:
    pair_sim = torch.diagonal(pair_sim)

train_pair["cosine"] = pair_sim.cpu().numpy()

embeddings_triplet_anc = torch.from_numpy(
    np.stack(train_triplet["anc_speech"].map(train_embeddings).to_numpy()).astype(np.float32)
)
embeddings_triplet_pos = torch.from_numpy(
    np.stack(train_triplet["pos_speech"].map(train_embeddings).to_numpy()).astype(np.float32)
)
embeddings_triplet_neg = torch.from_numpy(
    np.stack(train_triplet["neg_speech"].map(train_embeddings).to_numpy()).astype(np.float32)
)
triplet_sim_a = pairwise_cos_sim(embeddings_triplet_anc, embeddings_triplet_pos)
triplet_sim_b = pairwise_cos_sim(embeddings_triplet_anc, embeddings_triplet_neg)
if triplet_sim_a.ndim == 2:
    triplet_sim_a = torch.diagonal(triplet_sim_a)
if triplet_sim_b.ndim == 2:
    triplet_sim_b = torch.diagonal(triplet_sim_b)
triplet_sim = torch.min(triplet_sim_a, triplet_sim_b)

train_triplet["cosine"] = triplet_sim.cpu().numpy()

# ==== SAVING ====

train_pair.to_csv("./splits/train_paires.csv")
train_triplet.to_csv("./splits/train_triplets.csv")
dev_dataset_kl.to_csv("./splits/dev_kl.csv")
dev_dataset_spearman.to_csv("./splits/dev_spearman.csv")
test_dataset_kl.to_csv("./splits/test_kl.csv")
test_dataset_spearman.to_csv("./splits/test_spearman.csv")
