"""
Retrieve the most relevant evidence chunks for a researcher query.

This script loads the local TF-IDF index created by `embed.py`, transforms a
user query into the same vector space, and returns the highest-scoring chunks.

Input files:
    data/processed/<researcher_id>/tfidf_matrix.npz
    data/processed/<researcher_id>/tfidf_vectorizer.pkl
    data/processed/<researcher_id>/chunk_metadata.json

Example:
    python src/retrieve.py natalia_efremova "What are the main research interests?"
"""

import argparse
import json
import pickle
from pathlib import Path

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


PROCESSED_DIR = Path("data/processed")


def load_index(researcher_id):
    """
    Load the TF-IDF matrix, vectorizer, and chunk metadata for one researcher.

    Returns:
        A tuple of `(matrix, vectorizer, chunks)`.
    """
    researcher_dir = PROCESSED_DIR / researcher_id

    matrix_path = researcher_dir / "tfidf_matrix.npz"
    vectorizer_path = researcher_dir / "tfidf_vectorizer.pkl"
    metadata_path = researcher_dir / "chunk_metadata.json"

    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing matrix file: {matrix_path}")

    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Missing vectorizer file: {vectorizer_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    matrix = sparse.load_npz(matrix_path)

    with vectorizer_path.open("rb") as file:
        vectorizer = pickle.load(file)

    with metadata_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    return matrix, vectorizer, chunks


def retrieve(researcher_id, query, top_k=5):
    """
    Return the top matching chunks for a researcher query.

    Parameters:
        researcher_id: Folder name under `data/processed`.
        query: Natural-language search query.
        top_k: Number of chunks to return.

    Returns:
        List of result dictionaries containing score and chunk metadata.
    """
    matrix, vectorizer, chunks = load_index(researcher_id)

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()

    ranked_indexes = scores.argsort()[::-1][:top_k]

    results = []

    for rank, chunk_index in enumerate(ranked_indexes, start=1):
        chunk = chunks[chunk_index]

        results.append(
            {
                "rank": rank,
                "score": float(scores[chunk_index]),
                "chunk_id": chunk["chunk_id"],
                "source_title": chunk.get("source_title"),
                "source_file": chunk.get("source_file"),
                "source_type": chunk.get("source_type"),
                "text": chunk.get("text"),
            }
        )

    return results


def print_results(results):
    """
    Print retrieval results in a readable terminal format.
    """
    for result in results:
        print("=" * 80)
        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Source: {result['source_title']}")
        print(f"File: {result['source_file']}")
        print()
        print(result["text"][:1200])
        print()


def main():
    """
    Parse command-line arguments and run retrieval.
    """
    parser = argparse.ArgumentParser(
        description="Retrieve evidence chunks from a local researcher index."
    )
    parser.add_argument(
        "researcher_id",
        help="Folder name under data/processed, e.g. natalia_efremova",
    )
    parser.add_argument(
        "query",
        help="Question or search query, e.g. 'main research interests'",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to return.",
    )

    args = parser.parse_args()

    results = retrieve(args.researcher_id, args.query, top_k=args.top_k)
    print_results(results)


if __name__ == "__main__":
    main()