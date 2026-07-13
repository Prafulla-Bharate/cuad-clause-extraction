import argparse

import pandas as pd

from src.embeddings import ClauseIndex
from src.llm_client import GeminiClient


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic search over extracted clauses")
    parser.add_argument("--results", default="outputs/results.csv")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.results).fillna("")
    rows = df.to_dict(orient="records")

    client = GeminiClient()
    index = ClauseIndex(client)
    index.build(rows)

    hits = index.search(args.query, top_k=args.top_k)
    for hit in hits:
        print(f"\n[{hit['score']:.3f}] {hit['contract_id']} - {hit['clause_type']}")
        print(hit["text"][:300])


if __name__ == "__main__":
    main()
