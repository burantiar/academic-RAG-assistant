"""
Chunk processed researcher documents into citation-friendly passages.

This script reads normalized `.txt` files from:

    data/processed/<researcher_id>/

Each processed text file is expected to contain optional JSON metadata wrapped
in simple frontmatter:

    ---
    { "source_title": "Example", "source_type": "pdf" }
    ---
    Document text begins here...

The script writes one JSON Lines file:

    data/processed/<researcher_id>/chunks.jsonl

Each line is a chunk object containing source metadata plus a text passage.
These chunks will later be embedded and used for retrieval.
"""

import argparse
import json
from pathlib import Path


# Folder containing extracted text files.
PROCESSED_DIR = Path("data/processed")

# Approximate number of characters per chunk.
# Small enough for focused retrieval, large enough to preserve context.
CHUNK_SIZE = 1200

# Number of characters repeated between adjacent chunks.
# This helps avoid losing meaning at chunk boundaries.
CHUNK_OVERLAP = 200


def split_frontmatter(text):
    """
    Split optional JSON frontmatter from the document body.

    Parameters:
        text: Full text content of a processed `.txt` file.

    Returns:
        A tuple of `(metadata, body)`.

        metadata:
            Dictionary parsed from the JSON frontmatter, or `{}` if no valid
            frontmatter exists.

        body:
            The main document text.
    """
    if not text.startswith("---"):
        return {}, text

    # Split into:
    #   parts[0] = empty string before first ---
    #   parts[1] = metadata block
    #   parts[2] = remaining document body
    parts = text.split("---", 2)

    if len(parts) < 3:
        return {}, text

    metadata_text = parts[1].strip()
    body = parts[2].strip()

    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError:
        metadata = {}

    return metadata, body


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping character-based chunks.

    This is intentionally simple for the first version of the RAG pipeline.
    Later, we can replace it with token-aware or sentence-aware chunking.

    Parameters:
        text: Document body text to split.
        chunk_size: Maximum character length of each chunk.
        overlap: Number of characters repeated between consecutive chunks.

    Returns:
        A list of chunk strings.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward while keeping some overlap with the previous chunk.
        start = end - overlap

        # Defensive guard for unusual overlap/chunk size settings.
        if start < 0:
            start = 0

        if start >= len(text):
            break

    return chunks


def chunk_researcher(researcher_id):
    """
    Chunk all processed text files for a single researcher.

    Parameters:
        researcher_id:
            Folder name under `data/processed/`, for example
            `natalia_efremova`.

    Side effects:
        Writes `chunks.jsonl` into the researcher's processed folder.
    """
    researcher_dir = PROCESSED_DIR / researcher_id

    if not researcher_dir.exists():
        raise FileNotFoundError(f"Missing processed folder: {researcher_dir}")

    all_chunks = []

    for text_path in sorted(researcher_dir.glob("*.txt")):
        raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
        metadata, body = split_frontmatter(raw_text)
        chunks = chunk_text(body)

        for index, chunk in enumerate(chunks):
            # Stable chunk IDs help us trace retrieved evidence back to source.
            chunk_id = f"{researcher_id}:{text_path.stem}:{index:04d}"

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "researcher_id": researcher_id,
                    "source_file": metadata.get("source_file"),
                    "source_title": metadata.get("source_title", text_path.stem),
                    "source_type": metadata.get("source_type"),
                    "profile_url": metadata.get("profile_url"),
                    "chunk_index": index,
                    "text": chunk,
                }
            )

    output_path = researcher_dir / "chunks.jsonl"

    # JSON Lines format: one JSON object per line.
    # This is easy to stream, inspect, and append to later.
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_chunks)} chunks to {output_path}")


def main():
    """
    Parse command-line arguments and run chunking for one researcher.
    """
    parser = argparse.ArgumentParser(
        description="Chunk processed researcher documents into JSONL passages."
    )
    parser.add_argument(
        "researcher_id",
        help="Folder name under data/processed, e.g. natalia_efremova",
    )
    args = parser.parse_args()

    chunk_researcher(args.researcher_id)


if __name__ == "__main__":
    main()