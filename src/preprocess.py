from pathlib import Path
import re


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_gutenberg_text(text: str) -> str:
    """Clean Project Gutenberg text while preserving book content."""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # ---------------------------------------------------------
    # 1. Remove Gutenberg START header
    # ---------------------------------------------------------
    start = re.search(
        r"\*\*\*\s*START OF (?:THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if start:
        text = text[start.end():]

    # ---------------------------------------------------------
    # 2. Remove standard Gutenberg production information
    # ---------------------------------------------------------
    text = re.sub(
        r"Produced by.*?10,000 ebooks\.",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove single-line production credits
    text = re.sub(
        r"^Produced by [^\n]*\n?",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # ---------------------------------------------------------
    # 3. Remove 40190 transcriber's-notes box
    #
    # Find two horizontal borders and remove everything
    # between them.
    # ---------------------------------------------------------
    lines = text.splitlines()

    new_lines = []
    box_started = False

    for line in lines:
        stripped = line.strip()

        # Detect a border such as:
        # +-------------------------------------------------------------------+
        is_border = (
            stripped.startswith("+")
            and stripped.endswith("+")
            and "-" in stripped
        )

        if is_border and not box_started:
            box_started = True
            continue

        if is_border and box_started:
            box_started = False
            continue

        if box_started:
            continue

        new_lines.append(line)

    text = "\n".join(new_lines)
    # ---------------------------------------------------------
    # Remove remaining image/proofreading production credit
    # Used by book 56640.
    # ---------------------------------------------------------
    text = re.sub(
        r"Distributed Proofreading Team.*?"
        r"American Libraries\.\)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # ---------------------------------------------------------
    # 4. Remove 56640 UTF-8 transcriber note
    # ---------------------------------------------------------
    utf8_note = re.search(
        r"Transcriber[’']s Note:.*?"
        r"Additional notes are at the end of the book\.",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if utf8_note:
        text = (
            text[:utf8_note.start()]
            + text[utf8_note.end():]
        )

    # ---------------------------------------------------------
    # 5. Remove Gutenberg END section
    # ---------------------------------------------------------
    end = re.search(
        r"\*\*\*\s*END OF (?:THE )?PROJECT GUTENBERG EBOOK",
        text,
        flags=re.IGNORECASE,
    )

    if end:
        text = text[:end.start()]

    # ---------------------------------------------------------
    # 6. Normalize whitespace
    # ---------------------------------------------------------
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    return text.strip()


def process_book(path: Path):
    print(f"Processing: {path.name}")

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    cleaned = clean_gutenberg_text(text)

    output_path = PROCESSED_DIR / path.name

    output_path.write_text(
        cleaned,
        encoding="utf-8",
    )

    print(
        f"Saved: {output_path} "
        f"({len(cleaned):,} characters)"
    )


def main():
    files = sorted(RAW_DIR.glob("*.txt"))

    print(f"Found {len(files)} books.")

    for path in files:
        process_book(path)

    print("\nPreprocessing completed.")


if __name__ == "__main__":
    main()