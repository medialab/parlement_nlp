from sklearn.model_selection import train_test_split

from sentence_transformers.util import pairwise_cos_sim
from sentence_transformers import (
    SentenceTransformer
)
from transformers import AutoTokenizer

import torch
import numpy as np

import pandas as pd

SEED = 24

dataset = pd.read_csv("./data/paires-cosent.csv", dtype={"agreement": "float64"})
dataset = dataset[["a_speech", "b_speech", "agreement"]]
dataset = dataset.rename(columns={"agreement": "score"})

# ==== FILTERING TOKENS < 4096 ====

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
dataset["a_tokens"] = dataset["a_speech"].apply(lambda a: len(tokenizer(a)["input_ids"]))
dataset["b_tokens"] = dataset["b_speech"].apply(lambda a: len(tokenizer(a)["input_ids"]))

dataset["tokens"] = dataset[["a_tokens", "b_tokens"]].max(axis=1)

dataset = dataset[dataset["tokens"] < 4096]

del dataset["a_tokens"]
del dataset["b_tokens"]

# splitting

train, test = train_test_split(
    dataset, test_size=0.1, random_state=SEED, stratify=dataset["score"]
)
train, dev = train_test_split(
    train, test_size=0.1, random_state=SEED, stratify=train["score"]
)


# ==== DEV DATASETS ====

dev_sts = pd.read_parquet(
    "hf://datasets/CATIE-AQ/frenchSTS/data/validation-00000-of-00001.parquet"
)

# ==== TEST DATASETS ====

test_sts = pd.read_parquet(
    "hf://datasets/CATIE-AQ/frenchSTS/data/test-00000-of-00001.parquet"
)

# ==== COSINE SIM ====

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

train_texts = list(set(
    train["a_speech"].tolist()
    + train["b_speech"].tolist()
))

train_embeddings_array = model.encode(train_texts, batch_size=8, show_progress_bar=True)
train_embeddings = {text: embedding for text, embedding in zip(train_texts, train_embeddings_array)}

embeddings_a = torch.from_numpy(
    np.stack(train["a_speech"].map(train_embeddings).to_numpy()).astype(np.float32)
)
embeddings_b = torch.from_numpy(
    np.stack(train["b_speech"].map(train_embeddings).to_numpy()).astype(np.float32)
)
pair_sim = pairwise_cos_sim(embeddings_a, embeddings_b)
if pair_sim.ndim == 2:
    pair_sim = torch.diagonal(pair_sim)

train["cosine"] = pair_sim.cpu().numpy()

# ==== SAVING ====

train.to_csv("./splits/train_cosent.csv", index=None)
dev.to_csv("./splits/dev_cosent.csv", index=None)
test.to_csv("./splits/test_cosent.csv", index=None)
