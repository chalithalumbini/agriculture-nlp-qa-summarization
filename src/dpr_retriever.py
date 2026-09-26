from pathlib import Path
import json
import pickle

import faiss
import numpy as np
import torch

from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
)


PASSAGES_FILE = Path("data/passages.json")
INDEX_FILE = Path("data/dpr_passages.faiss")
METADATA_FILE = Path("data/dpr_metadata.pkl")


QUESTION_MODEL = "facebook/dpr-question_encoder-single-nq-base"
CONTEXT_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"


def load_passages():
    """Load passages from JSON."""

    with open(PASSAGES_FILE, "r", encoding="utf-8") as f:
        passages = json.load(f)

    print(f"Loaded {len(passages)} passages.")

    return passages


def build_passage_embeddings(passages, tokenizer, encoder):
    """Convert passages into DPR embeddings."""

    embeddings = []

    batch_size = 8

    for start in range(0, len(passages), batch_size):

        batch = passages[start:start + batch_size]

        texts = [
            passage["text"]
            for passage in batch
        ]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        with torch.no_grad():

            outputs = encoder(**inputs)

        batch_embeddings = outputs.pooler_output.cpu().numpy()

        embeddings.append(batch_embeddings)

        print(
            f"Encoded {min(start + batch_size, len(passages))}"
            f"/{len(passages)}"
        )

    return np.vstack(embeddings).astype("float32")


def main():

    print("Loading passages...")

    passages = load_passages()


    # -----------------------------------------------------
    # Load DPR context encoder
    # -----------------------------------------------------

    print("\nLoading DPR context tokenizer...")

    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        CONTEXT_MODEL
    )

    print("Loading DPR context encoder...")

    context_encoder = DPRContextEncoder.from_pretrained(
        CONTEXT_MODEL
    )

    context_encoder.eval()


    # -----------------------------------------------------
    # Encode passages
    # -----------------------------------------------------

    print("\nEncoding passages...")

    embeddings = build_passage_embeddings(
        passages,
        context_tokenizer,
        context_encoder,
    )

    print("\nEmbedding shape:", embeddings.shape)


    # -----------------------------------------------------
    # Build FAISS index
    # -----------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    # Normalize vectors so inner product behaves like
    # cosine similarity.
    faiss.normalize_L2(embeddings)

    index.add(embeddings)

    print("FAISS index size:", index.ntotal)


    # -----------------------------------------------------
    # Save FAISS index
    # -----------------------------------------------------

    faiss.write_index(
        index,
        str(INDEX_FILE),
    )


    # -----------------------------------------------------
    # Save passage metadata
    # -----------------------------------------------------

    with open(METADATA_FILE, "wb") as f:

        pickle.dump(
            passages,
            f,
        )


    print("\nDPR passage index created successfully.")

    print(f"Index:    {INDEX_FILE}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()