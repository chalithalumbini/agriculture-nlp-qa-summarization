import pickle

import faiss
import numpy as np
import torch

from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)


INDEX_FILE = "data/dpr_passages.faiss"
METADATA_FILE = "data/dpr_metadata.pkl"

QUESTION_MODEL = "facebook/dpr-question_encoder-single-nq-base"


def main():

    # -----------------------------------------------------
    # Load FAISS index
    # -----------------------------------------------------

    print("Loading FAISS index...")

    index = faiss.read_index(INDEX_FILE)


    # -----------------------------------------------------
    # Load passage metadata
    # -----------------------------------------------------

    print("Loading passage metadata...")

    with open(METADATA_FILE, "rb") as f:
        passages = pickle.load(f)


    # -----------------------------------------------------
    # Load DPR question encoder
    # -----------------------------------------------------

    print("Loading DPR question tokenizer...")

    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        QUESTION_MODEL
    )

    print("Loading DPR question encoder...")

    encoder = DPRQuestionEncoder.from_pretrained(
        QUESTION_MODEL
    )

    encoder.eval()


    # -----------------------------------------------------
    # Ask a test question
    # -----------------------------------------------------

    question = "How can soil fertility be maintained?"

    print("\nQuestion:")
    print(question)


    # -----------------------------------------------------
    # Convert question into a DPR vector
    # -----------------------------------------------------

    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():

        output = encoder(**inputs)

    question_embedding = output.pooler_output.cpu().numpy()


    # -----------------------------------------------------
    # Normalize question vector
    # -----------------------------------------------------

    question_embedding = question_embedding.astype("float32")

    faiss.normalize_L2(question_embedding)


    # -----------------------------------------------------
    # Search FAISS
    # -----------------------------------------------------

    top_k = 5

    scores, indices = index.search(
        question_embedding,
        top_k,
    )


    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\nTop retrieved passages:")
    print("=" * 80)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):

        passage = passages[idx]

        print(f"\nRank: {rank}")
        print(f"Score: {score:.4f}")
        print(f"Passage ID: {passage['passage_id']}")
        print(f"Book ID: {passage['book_id']}")
        print(f"Book: {passage['book_title']}")
        print("\nText:")
        print(passage["text"][:1000])
        print("-" * 80)


if __name__ == "__main__":
    main()