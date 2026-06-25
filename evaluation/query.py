import pandas as pd
from ast import literal_eval
from sentence_transformers.util import semantic_search
import argparse
import numpy as np

import chromadb

chroma_client = chromadb.Client()


def find_top_k():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        help="CSV path for parlement embeddings",
        default=None,
    )
    args = parser.parse_args()

    df = pd.read_csv(
        args.csv,
        converters={
            "embedding_a_speech": literal_eval,
            "embedding_b_speech": literal_eval,
        },
    )

    print(df)

    part_a = df[["a_speech", "embedding_a_speech"]]
    part_a = part_a.rename(
        columns={"a_speech": "speech", "embedding_a_speech": "embedding"}
    )

    part_b = df[["b_speech", "embedding_b_speech"]]
    part_b = part_b.rename(
        columns={"b_speech": "speech", "embedding_b_speech": "embedding"}
    )

    db = pd.concat([part_a, part_b]).drop_duplicates(subset=["speech"])

    queries = db.sample(n=10)

    collection = chroma_client.create_collection(name="collection")
    collection.add(
        ids=db.index.tolist(),
        documents=db["speech"].tolist(),
        embeddings=db["embedding"].tolist()
    )

