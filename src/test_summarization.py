from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


MODEL_NAME = "t5-small"


def main():

    print("Loading T5 tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading T5 model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    text = (
        "Soil fertility is important for successful crop production. "
        "When land becomes worn out, it may have lost important plant "
        "nutrients such as nitrogen, phosphoric acid, and potash. "
        "Farmers can restore soil fertility by returning the missing "
        "plant foods to the soil. Crop rotation can also help maintain "
        "fertility by preventing the repeated growth of the same crop "
        "and improving the condition of the soil."
    )

    # T5 uses a task prefix.
    input_text = "summarize: " + text

    print("\nInput text:")
    print(text)

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    print("\nGenerating summary...")

    output_ids = model.generate(
        **inputs,
        max_length=80,
        min_length=20,
        num_beams=4,
        early_stopping=True,
    )

    summary = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()