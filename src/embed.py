"""
Build a local searchable index over researcher text chunks.

This first version uses TF-IDF vectors from scikit-learn. It is not as
semantically powerful as modern embedding models, but it is simple, local,
cheap, and excellent for testing the retrieval pipeline.

Input:
    data/processed/<researcher_id>/chunks.jsonl

Outputs:
    data/processed/<researcher_id>/tfidf_matrix.npz
    data/processed/<researcher_id>/tfidf_vectorizer.pkl
    data/processed/<researcher_id>/chunk_metadata.json
"""

import argparse
import json
import pickle
from pathlib import Path

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


PROCESSED_DIR = Path("data/processed")


def load_chunks(chunks_path):
    """
    Load chunks from a JSON Lines file.

    Each line should be one JSON object with at least:
        - chunk_id
        - text
        - source_title
        - source_file
    """
    chunks = []

    with chunks_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            chunks.append(json.loads(line))

    return chunks


def build_index(researcher_id):
    """
    Build and save a TF-IDF index for one researcher.
    """
    researcher_dir = PROCESSED_DIR / researcher_id
    chunks_path = researcher_dir / "chunks.jsonl"

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    chunks = load_chunks(chunks_path)

    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    texts = [chunk["text"] for chunk in chunks]

    # TF-IDF turns each chunk into a sparse vector of weighted terms.
    # ngram_range=(1, 2) includes single words and short two-word phrases.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
    )

    matrix = vectorizer.fit_transform(texts)

    matrix_path = researcher_dir / "tfidf_matrix.npz"
    vectorizer_path = researcher_dir / "tfidf_vectorizer.pkl"
    metadata_path = researcher_dir / "chunk_metadata.json"

    sparse.save_npz(matrix_path, matrix)

    with vectorizer_path.open("wb") as file:
        pickle.dump(vectorizer, file)

    # Store chunk metadata separately from the matrix so retrieval can return
    # citations and source details alongside matching text.
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print(f"Indexed {len(chunks)} chunks for {researcher_id}")
    print(f"Wrote matrix to {matrix_path}")
    print(f"Wrote vectorizer to {vectorizer_path}")
    print(f"Wrote metadata to {metadata_path}")


def main():
    """
    Parse command-line arguments and build the researcher index.
    """
    parser = argparse.ArgumentParser(
        description="Build a local TF-IDF index over researcher chunks."
    )
    parser.add_argument(
        "researcher_id",
        help="Folder name under data/processed, e.g. natalia_efremova",
    )
    args = parser.parse_args()

    build_index(args.researcher_id)


if __name__ == "__main__":
    main()