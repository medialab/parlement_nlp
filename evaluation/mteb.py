import argparse

import mteb
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to the SentenceTransformer model")
    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    
    tasks = mteb.get_tasks(languages=["fr"])
    evaluation = mteb.evaluate(tasks=tasks)

    results = evaluation.run(model, output_folder=f"mtep_results/{args.model}")