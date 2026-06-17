import argparse
import json
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def clean_text(text):
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def extract_pdf_text(path):
    reader = PdfReader(str(path))
    pages = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n\n[Page {index}]\n{text}")

    return clean_text("\n".join(pages))


def extract_html_text(path):
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else path.stem
    text = soup.get_text("\n")

    return title, clean_text(text)


def write_text(output_path, metadata, text):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = [
        "---",
        json.dumps(metadata, indent=2),
        "---",
        text,
    ]

    output_path.write_text("\n".join(content), encoding="utf-8")


def extract_researcher(researcher_id):
    researcher_raw_dir = RAW_DIR / researcher_id
    researcher_processed_dir = PROCESSED_DIR / researcher_id

    if not researcher_raw_dir.exists():
        raise FileNotFoundError(f"Missing researcher folder: {researcher_raw_dir}")

    metadata_path = researcher_raw_dir / "metadata.json"
    researcher_metadata = {}

    if metadata_path.exists():
        researcher_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    pdf_dir = researcher_raw_dir / "pdfs"
    webpage_dir = researcher_raw_dir / "webpages"

    extracted_count = 0

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        text = extract_pdf_text(pdf_path)
        metadata = {
            **researcher_metadata,
            "source_type": "pdf",
            "source_file": str(pdf_path),
            "source_title": pdf_path.stem,
        }

        output_path = researcher_processed_dir / f"{pdf_path.stem}.txt"
        write_text(output_path, metadata, text)
        extracted_count += 1

    for html_path in sorted(webpage_dir.glob("*")):
        if html_path.suffix.lower() not in {".html", ".htm", ".txt"}:
            continue

        if html_path.suffix.lower() == ".txt":
            title = html_path.stem
            text = clean_text(html_path.read_text(encoding="utf-8", errors="ignore"))
        else:
            title, text = extract_html_text(html_path)

        metadata = {
            **researcher_metadata,
            "source_type": "webpage",
            "source_file": str(html_path),
            "source_title": title,
        }

        output_path = researcher_processed_dir / f"{html_path.stem}.txt"
        write_text(output_path, metadata, text)
        extracted_count += 1

    print(f"Extracted {extracted_count} files into {researcher_processed_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("researcher_id", help="Folder name under data/raw, e.g. prof_a")
    args = parser.parse_args()

    extract_researcher(args.researcher_id)


if __name__ == "__main__":
    main()