import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)


MODEL_NAME = "deepset/bert-base-cased-squad2"


def main():

    print("Loading BERT-QA tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading BERT-QA model...")

    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    question = (
        "How can worn-out land be made fertile again?"
    )

    context = (
        "Worn out land usually means that a soil has been "
        "robbed of one of these plant necessities, or of two "
        "or of all three. To make the land once more fruitful "
        "it is necessary to restore the missing food or foods. "
        "How can this be done? Two of these plant foods, "
        "namely, phosphoric acid and potash, are minerals."
    )

    print("\nQuestion:")
    print(question)

    print("\nContext:")
    print(context)

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
    )

    with torch.no_grad():

        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_position = torch.argmax(start_logits, dim=1).item()
    end_position = torch.argmax(end_logits, dim=1).item()

    if end_position < start_position:
        print("\nCould not find a valid answer span.")
        return

    answer_tokens = inputs["input_ids"][
        0,
        start_position:end_position + 1
    ]

    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True,
    )

    start_score = torch.softmax(
        start_logits, dim=1
    )[0, start_position].item()

    end_score = torch.softmax(
        end_logits, dim=1
    )[0, end_position].item()

    confidence = (start_score + end_score) / 2

    print("\nAnswer:")
    print(answer)

    print("\nConfidence:")
    print(confidence)


if __name__ == "__main__":
    main()