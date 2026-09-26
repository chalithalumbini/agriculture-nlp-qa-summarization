import pickle

import faiss
import torch

from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


# ---------------------------------------------------------
# Files and models
# ---------------------------------------------------------

INDEX_FILE = "data/dpr_passages.faiss"
METADATA_FILE = "data/dpr_metadata.pkl"

DPR_MODEL = "facebook/dpr-question_encoder-single-nq-base"
SUMMARY_MODEL = "t5-small"


# ---------------------------------------------------------
# Load DPR
# ---------------------------------------------------------

print("Loading DPR question encoder...")

dpr_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    DPR_MODEL
)

dpr_encoder = DPRQuestionEncoder.from_pretrained(
    DPR_MODEL
)

dpr_encoder.eval()


# ---------------------------------------------------------
# Load T5
# ---------------------------------------------------------

print("Loading T5 summarization model...")

summary_tokenizer = AutoTokenizer.from_pretrained(
    SUMMARY_MODEL
)

summary_model = AutoModelForSeq2SeqLM.from_pretrained(
    SUMMARY_MODEL
)

summary_model.eval()


# ---------------------------------------------------------
# Load FAISS index and passages
# ---------------------------------------------------------

print("Loading DPR index...")

index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "rb") as f:
    passages = pickle.load(f)

print(f"Loaded {len(passages)} passages.")


# ---------------------------------------------------------
# DPR retrieval
# ---------------------------------------------------------

def retrieve_passages(topic, top_k=2):

    inputs = dpr_tokenizer(
        topic,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():

        output = dpr_encoder(**inputs)

    topic_embedding = (
        output.pooler_output
        .cpu()
        .numpy()
        .astype("float32")
    )

    faiss.normalize_L2(topic_embedding)

    scores, indices = index.search(
        topic_embedding,
        top_k,
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        results.append(
            {
                "dpr_score": float(score),
                "passage": passages[idx],
            }
        )

    return results


# ---------------------------------------------------------
# T5 summarization
# ---------------------------------------------------------

def summarize_text(text, max_length=80):

    input_text = "summarize: " + text

    inputs = summary_tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    with torch.no_grad():

        output_ids = summary_model.generate(
            **inputs,
            max_length=max_length,
            min_length=15,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return summary_tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )


# ---------------------------------------------------------
# Complete DPR + T5 summarization system
# ---------------------------------------------------------

def summarize_topic(topic):

    print("\n" + "=" * 80)

    print("TOPIC:")
    print(topic)

    # -----------------------------------------------------
    # Step 1: Retrieve relevant passages with DPR
    # -----------------------------------------------------

    print("\nRetrieving relevant passages with DPR...")

    retrieved = retrieve_passages(
        topic,
        top_k=2,
    )

    print("\nRetrieved passages:")
    print("-" * 80)

    for rank, item in enumerate(
        retrieved,
        start=1,
    ):

        passage = item["passage"]

        print(
            f"{rank}. "
            f"{passage['passage_id']} "
            f"(DPR score: {item['dpr_score']:.4f})"
        )

        print(
            f"   Book: {passage['book_title']}"
        )

    # -----------------------------------------------------
    # Step 2: Summarize each retrieved passage
    # -----------------------------------------------------

    print("\nGenerating passage summaries...")

    passage_summaries = []

    for rank, item in enumerate(
        retrieved,
        start=1,
    ):

        passage = item["passage"]

        summary = summarize_text(
            passage["text"],
            max_length=80,
        )

        passage_summaries.append(summary)

        print(f"\nSummary {rank}:")
        print(summary)

    # -----------------------------------------------------
    # Step 3: Combine the generated summaries
    # -----------------------------------------------------

    combined_summaries = " ".join(
        passage_summaries
    )

    # -----------------------------------------------------
    # Step 4: Generate final topic summary
    # -----------------------------------------------------

    print("\nGenerating final topic summary...")

    final_summary = summarize_text(
        combined_summaries,
        max_length=100,
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n" + "=" * 80)

    print("TOPIC-FOCUSED SUMMARY:")
    print(final_summary)

    print("\nSources:")

    for item in retrieved:

        passage = item["passage"]

        print(
            f"- {passage['passage_id']} "
            f"({passage['book_title']})"
        )

    print("=" * 80)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    topic = (
        "soil fertility and how farmers can maintain "
        "fertile soil"
    )

    summarize_topic(topic)