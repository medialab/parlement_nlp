import pandas as pd
from ast import literal_eval
import argparse
import chromadb
import uuid
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

chroma_client = chromadb.Client()


def upsert_in_batches(collection, db, batch_size):
    ids = db["ids"].tolist()
    documents = db["amendment_summary"].tolist()
    embeddings = db["embedding_amendment_summary"].tolist()
    groups = db["author_group"].tolist()

    for start in range(0, len(db), batch_size):
        end = min(start + batch_size, len(db))
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=[{"group": g} for g in groups],
        )


def find_top_k(collection_pre, collection_post, queries, k):
    for text in queries:
        # Ask for one extra result because the query itself is in the DB.
        results_pre = collection_pre.query(
            query_texts=[text],
            n_results=k + 1,
        )
        results_post = collection_post.query(
            query_texts=[text],
            n_results=k + 1,
        )

        pre_ids = results_pre["ids"][0]
        pre_docs = results_pre["documents"][0]
        pre_distances = results_pre["distances"][0]
        pre_groups = [m["group"] for m in results_pre["metadatas"][0]]

        post_ids = results_post["ids"][0]
        post_docs = results_post["documents"][0]
        post_distances = results_post["distances"][0]
        post_groups = [m["group"] for m in results_post["metadatas"][0]]

        pre_rank = { i:r for i, r in zip(ids, range(len(pre_ids))) }

        for i, (id, doc, dist, group) in enumerate(zip(post_ids, post_docs, post_distances, post_groups)):
            rank_origin = pre_rank[id]
            evol = rank_origin - i
            evol = f"+{evol}" if evol > 0 else evol
            print(f"\t[#{rank_origin} -> #{i}; group=\"{group}\"; evolution={evol}]")

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_pre_train",
        help="CSV path for parlement embeddings pre-train",
        default=None,
    )
    parser.add_argument(
        "csv_post_train",
        help="CSV path for parlement embeddings post-train",
        default=None,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of closest speeches to print for each query",
    )
    parser.add_argument(
        "--model-pre",
        help="Model to use",
    )
    parser.add_argument(
        "--model-post",
        help="Model to use",
    )
    args = parser.parse_args()

    df_pre = pd.read_csv(
        args.csv_pre_train,
        converters={
            "embedding_amendment_summary": literal_eval,
        },
    )

    df_post = pd.read_csv(
        args.csv_post_train,
        converters={
            "embedding_amendment_summary": literal_eval,
        },
    )

    ids = [str(uuid.uuid1()) for i in df_pre.index.tolist()]
    df_pre["ids"] = ids
    df_post["ids"] = ids

    sentence_transformer_ef_pre = SentenceTransformerEmbeddingFunction(
        model_name=args.model_pre, device="mps", normalize_embeddings=False
    )

    sentence_transformer_ef_post = SentenceTransformerEmbeddingFunction(
        model_name=args.model_post, device="mps", normalize_embeddings=False
    )

    collection_qwen = chroma_client.get_or_create_collection(
        name="qwen", embedding_function=sentence_transformer_ef_pre
    )
    collection_finetuned = chroma_client.get_or_create_collection(
        name="finetuned", embedding_function=sentence_transformer_ef_post
    )

    max_batch_size = chroma_client.get_max_batch_size()
    safe_batch_size = max(1, min(1000, max_batch_size))

    upsert_in_batches(collection=collection_finetuned, db=df_pre, batch_size=safe_batch_size)
    upsert_in_batches(collection=collection_qwen, db=df_post, batch_size=safe_batch_size)

    while True:
        query = [input("Query : ")]
        find_top_k(collection_qwen, collection_finetuned, queries=query, k=args.k)
