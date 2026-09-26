from pathlib import Path
import json


PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = Path("data/passages.json")


PASSAGE_SIZE = 200
OVERLAP = 50
STEP = PASSAGE_SIZE - OVERLAP


def create_passages(text: str):
    """
    Create overlapping passages using word windows.

    Each passage contains approximately 200 words.
    Consecutive passages overlap by 50 words.
    """

    words = text.split()

    passages = []

    start = 0

    while start < len(words):

        end = start + PASSAGE_SIZE

        passage_words = words[start:end]

        if not passage_words:
            break

        passages.append(" ".join(passage_words))

        # Move forward while keeping overlap
        start += STEP

    return passages


def process_book(path: Path):

    print(f"Processing: {path.name}")

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    passages = create_passages(text)

    book_id = path.stem

    results = []

    for i, passage in enumerate(passages):

        results.append(
            {
                "passage_id": f"{book_id}_{i:04d}",
                "book_id": book_id,
                "book_title": book_id,
                "text": passage,
            }
        )

    print(f"  Words:    {len(text.split()):,}")
    print(f"  Passages: {len(results)}")

    return results


def main():

    files = sorted(PROCESSED_DIR.glob("*.txt"))

    print(f"Found {len(files)} processed books.\n")

    all_passages = []

    for path in files:

        passages = process_book(path)

        all_passages.extend(passages)

    OUTPUT_FILE.write_text(
        json.dumps(
            all_passages,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n--------------------------------")
    print(f"Total passages: {len(all_passages)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("--------------------------------")


if __name__ == "__main__":
    main()