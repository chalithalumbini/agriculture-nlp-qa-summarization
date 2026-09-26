import pickle

import faiss
import torch

from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)


INDEX_FILE = "data/dpr_passages.faiss"
METADATA_FILE = "data/dpr_metadata.pkl"

DPR_MODEL = "facebook/dpr-question_encoder-single-nq-base"
QA_MODEL = "deepset/bert-base-cased-squad2"


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
# Load BERT-QA
# ---------------------------------------------------------

print("Loading BERT-QA model...")

qa_tokenizer = AutoTokenizer.from_pretrained(
    QA_MODEL
)

qa_model = AutoModelForQuestionAnswering.from_pretrained(
    QA_MODEL
)

qa_model.eval()


# ---------------------------------------------------------
# Load FAISS index
# ---------------------------------------------------------

print("Loading DPR index...")

index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "rb") as f:
    passages = pickle.load(f)

print(f"Loaded {len(passages)} passages.")


# ---------------------------------------------------------
# DPR retrieval
# ---------------------------------------------------------

def retrieve_passages(question, top_k=5):

    inputs = dpr_tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        output = dpr_encoder(**inputs)

    question_embedding = (
        output.pooler_output
        .cpu()
        .numpy()
        .astype("float32")
    )

    faiss.normalize_L2(question_embedding)

    scores, indices = index.search(
        question_embedding,
        top_k,
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        results.append(
            {
                "dpr_score": float(score),
                "passage": passages[idx],
            }
        )

    return results


# ---------------------------------------------------------
# BERT-QA
# ---------------------------------------------------------

def answer_question(question, context):

    inputs = qa_tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = qa_model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # Convert logits to probabilities
    start_probs = torch.softmax(
        start_logits,
        dim=0,
    )

    end_probs = torch.softmax(
        end_logits,
        dim=0,
    )

    best_answer = None
    best_score = 0.0

    # Look for valid answer spans.
    for start in range(len(start_logits)):

        for end in range(
            start,
            min(start + 30, len(end_logits)),
        ):

            # Avoid invalid spans
            if end < start:
                continue

            answer_tokens = inputs["input_ids"][
                0,
                start:end + 1,
            ]

            answer = qa_tokenizer.decode(
                answer_tokens,
                skip_special_tokens=True,
            ).strip()

            if not answer:
                continue

            # Ignore special tokens
            if answer in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            score = (
                start_probs[start].item()
                * end_probs[end].item()
            )

            if score > best_score:

                best_score = score
                best_answer = answer

    return best_answer, best_score


# ---------------------------------------------------------
# Complete QA system
# ---------------------------------------------------------

def ask(question):

    print("\n" + "=" * 80)
    print("QUESTION:")
    print(question)

    # -----------------------------------------------------
    # Step 1: DPR
    # -----------------------------------------------------

    print("\nRetrieving passages with DPR...")

    retrieved = retrieve_passages(
        question,
        top_k=5,
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

    # -----------------------------------------------------
    # Step 2: BERT-QA
    # -----------------------------------------------------

    print("\nRunning BERT-QA...")

    candidates = []

    for item in retrieved:

        passage = item["passage"]

        answer, qa_score = answer_question(
            question,
            passage["text"],
        )

        print(
            f"\nPassage: {passage['passage_id']}"
        )

        print(
            f"DPR score: {item['dpr_score']:.4f}"
        )

        print(
            f"BERT-QA score: {qa_score:.4f}"
        )

        print(
            f"Answer: {answer}"
        )

        if answer is not None:

            candidates.append(
                {
                    "answer": answer,
                    "qa_score": qa_score,
                    "dpr_score": item["dpr_score"],
                    "passage": passage,
                }
            )

    # -----------------------------------------------------
    # Step 3: Select answer
    # -----------------------------------------------------

    if not candidates:

        print("\nNo answer could be extracted.")
        return

    best = max(
        candidates,
        key=lambda x: x["qa_score"],
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n" + "=" * 80)

    print("FINAL ANSWER:")
    print(best["answer"])

    print("\nBERT-QA score:")
    print(f"{best['qa_score']:.4f}")

    print("\nDPR score:")
    print(f"{best['dpr_score']:.4f}")

    print("\nSource passage:")
    print(best["passage"]["passage_id"])

    print("\nBook:")
    print(best["passage"]["book_title"])

    print("=" * 80)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    question = (
        "How can soil fertility be maintained?"
    )

    ask(question)