import pickle

import faiss
import streamlit as st
import torch

from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INDEX_FILE = "data/dpr_passages.faiss"
METADATA_FILE = "data/dpr_metadata.pkl"

DPR_MODEL = "facebook/dpr-question_encoder-single-nq-base"
QA_MODEL = "deepset/bert-base-cased-squad2"
SUMMARY_MODEL = "t5-small"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Agriculture NLP System",
    page_icon="🌱",
    layout="wide",
)


st.title("🌱 Agriculture NLP QA & Summarization")

st.write(
    "Ask questions about the agriculture books or "
    "enter a topic to generate a summary."
)


# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------

@st.cache_resource
def load_models():

    # DPR
    dpr_tokenizer = (
        DPRQuestionEncoderTokenizer.from_pretrained(
            DPR_MODEL
        )
    )

    dpr_encoder = (
        DPRQuestionEncoder.from_pretrained(
            DPR_MODEL
        )
    )

    dpr_encoder.eval()

    # BERT QA
    qa_tokenizer = AutoTokenizer.from_pretrained(
        QA_MODEL
    )

    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        QA_MODEL
    )

    qa_model.eval()

    # T5
    summary_tokenizer = AutoTokenizer.from_pretrained(
        SUMMARY_MODEL
    )

    summary_model = AutoModelForSeq2SeqLM.from_pretrained(
        SUMMARY_MODEL
    )

    summary_model.eval()

    return (
        dpr_tokenizer,
        dpr_encoder,
        qa_tokenizer,
        qa_model,
        summary_tokenizer,
        summary_model,
    )


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

@st.cache_resource
def load_data():

    index = faiss.read_index(
        INDEX_FILE
    )

    with open(
        METADATA_FILE,
        "rb",
    ) as f:

        passages = pickle.load(f)

    return index, passages


with st.spinner("Loading models..."):

    (
        dpr_tokenizer,
        dpr_encoder,
        qa_tokenizer,
        qa_model,
        summary_tokenizer,
        summary_model,
    ) = load_models()


index, passages = load_data()


# ---------------------------------------------------------
# DPR retrieval
# ---------------------------------------------------------

def retrieve_passages(
    question,
    top_k=3,
):

    inputs = dpr_tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():

        output = dpr_encoder(**inputs)

    embedding = (
        output.pooler_output
        .cpu()
        .numpy()
        .astype("float32")
    )

    faiss.normalize_L2(
        embedding
    )

    scores, indices = index.search(
        embedding,
        top_k,
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        results.append(
            {
                "score": float(score),
                "passage": passages[idx],
            }
        )

    return results


# ---------------------------------------------------------
# BERT-QA
# ---------------------------------------------------------

def answer_question(
    question,
    context,
):

    inputs = qa_tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():

        outputs = qa_model(
            **inputs
        )

    start_probs = torch.softmax(
        outputs.start_logits[0],
        dim=0,
    )

    end_probs = torch.softmax(
        outputs.end_logits[0],
        dim=0,
    )

    best_answer = None
    best_score = 0.0

    for start in range(
        len(start_probs)
    ):

        for end in range(
            start,
            min(
                start + 30,
                len(end_probs),
            ),
        ):

            answer_tokens = inputs[
                "input_ids"
            ][
                0,
                start:end + 1,
            ]

            answer = qa_tokenizer.decode(
                answer_tokens,
                skip_special_tokens=True,
            ).strip()

            if not answer:
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
# T5 summarization
# ---------------------------------------------------------

def summarize_text(
    text,
    max_length=100,
):

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
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("System")

mode = st.sidebar.radio(
    "Select mode:",
    [
        "Question Answering",
        "Topic Summarization",
    ],
)

st.sidebar.write(
    f"Loaded {len(passages):,} agriculture passages."
)


# ---------------------------------------------------------
# Question Answering
# ---------------------------------------------------------

if mode == "Question Answering":

    st.header("Question Answering")

    question = st.text_input(
        "Enter your question:",
        placeholder=(
            "How can soil fertility be maintained?"
        ),
    )

    if st.button(
        "Answer Question",
        type="primary",
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Retrieving and answering..."
            ):

                retrieved = retrieve_passages(
                    question,
                    top_k=5,
                )

                candidates = []

                for item in retrieved:

                    passage = item["passage"]

                    answer, qa_score = answer_question(
                        question,
                        passage["text"],
                    )

                    if answer:

                        candidates.append(
                            {
                                "answer": answer,
                                "qa_score": qa_score,
                                "dpr_score": item["score"],
                                "passage": passage,
                            }
                        )

                if not candidates:

                    st.warning(
                        "No answer could be extracted."
                    )

                else:

                    best = max(
                        candidates,
                        key=lambda x: x["qa_score"],
                    )

                    st.subheader("Answer")

                    st.success(
                        best["answer"]
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "BERT-QA Score",
                            f"{best['qa_score']:.4f}",
                        )

                    with col2:

                        st.metric(
                            "DPR Score",
                            f"{best['dpr_score']:.4f}",
                        )

                    st.subheader(
                        "Source"
                    )

                    st.write(
                        f"**Passage:** "
                        f"{best['passage']['passage_id']}"
                    )

                    st.write(
                        f"**Book:** "
                        f"{best['passage']['book_title']}"
                    )

                    with st.expander(
                        "Show source passage"
                    ):

                        st.write(
                            best["passage"]["text"]
                        )

                    st.subheader(
                        "Retrieved Passages"
                    )

                    for rank, item in enumerate(
                        retrieved,
                        start=1,
                    ):

                        passage = item["passage"]

                        with st.expander(
                            f"{rank}. "
                            f"{passage['passage_id']} "
                            f"— DPR "
                            f"{item['score']:.4f}"
                        ):

                            st.write(
                                passage["text"]
                            )


# ---------------------------------------------------------
# Topic Summarization
# ---------------------------------------------------------

else:

    st.header("Topic Summarization")

    topic = st.text_input(
        "Enter a topic:",
        placeholder=(
            "soil fertility and how farmers "
            "can maintain fertile soil"
        ),
    )

    if st.button(
        "Summarize Topic",
        type="primary",
    ):

        if not topic.strip():

            st.warning(
                "Please enter a topic."
            )

        else:

            with st.spinner(
                "Retrieving passages and generating summary..."
            ):

                retrieved = retrieve_passages(
                    topic,
                    top_k=2,
                )

                passage_summaries = []

                for item in retrieved:

                    passage = item["passage"]

                    summary = summarize_text(
                        passage["text"],
                        max_length=80,
                    )

                    passage_summaries.append(
                        summary
                    )

                combined_summaries = " ".join(
                    passage_summaries
                )

                final_summary = summarize_text(
                    combined_summaries,
                    max_length=100,
                )

            st.subheader(
                "Topic-Focused Summary"
            )

            st.info(
                final_summary
            )

            st.subheader(
                "Retrieved Sources"
            )

            for rank, item in enumerate(
                retrieved,
                start=1,
            ):

                passage = item["passage"]

                with st.expander(
                    f"{rank}. "
                    f"{passage['passage_id']} "
                    f"— DPR "
                    f"{item['score']:.4f}"
                ):

                    st.write(
                        f"**Book:** "
                        f"{passage['book_title']}"
                    )

                    st.write(
                        passage["text"]
                    )