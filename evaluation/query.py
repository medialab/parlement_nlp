import pandas as pd
from ast import literal_eval
import argparse
import chromadb
import uuid
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

chroma_client = chromadb.Client()


def upsert_in_batches(collection, db, batch_size):
    ids = [str(uuid.uuid1()) for i in db.index.tolist()]
    documents = db["speech"].tolist()
    embeddings = db["embedding"].tolist()

    for start in range(0, len(db), batch_size):
        end = min(start + batch_size, len(db))
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
        )


def find_top_k(collection, queries, k):
    for text in queries:

        # Ask for one extra result because the query itself is in the DB.
        results = collection.query(
            query_texts=[text],
            n_results=k + 1,
        )

        docs = results["documents"][0]
        distances = results["distances"][0]

        print("=" * 120)
        print(f"Query: {query}")
        print(f"Top {k} closest speeches:")
        for rank, (doc_text, distance) in enumerate(zip(docs, distances)):
            print(f"  {rank}. [distance={distance:.6f}] {doc_text}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        help="CSV path for parlement embeddings",
        default=None,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of closest speeches to print for each query",
    )
    parser.add_argument(
        "--model",
        default=5,
        help="Model to use",
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

    sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
        model_name=args.model,
        device="mps",
        normalize_embeddings=False
    )

    part_a = df[["a_speech", "embedding_a_speech"]]
    part_a = part_a.rename(
        columns={"a_speech": "speech", "embedding_a_speech": "embedding"}
    )

    part_b = df[["b_speech", "embedding_b_speech"]]
    part_b = part_b.rename(
        columns={"b_speech": "speech", "embedding_b_speech": "embedding"}
    )

    db = pd.concat([part_a, part_b]).drop_duplicates(subset=["speech"])
    db = db.reset_index(drop=True)

    queries = db.sample(n=10)

    collection = chroma_client.get_or_create_collection(name="collection", embedding_function=sentence_transformer_ef)

    max_batch_size = chroma_client.get_max_batch_size()
    safe_batch_size = max(1, min(1000, max_batch_size))
    upsert_in_batches(collection=collection, db=db, batch_size=safe_batch_size)

    while True:
        query = [input("Query : ")]
        find_top_k(collection=collection, queries=query, k=args.k)

