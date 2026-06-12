import os
import wandb
from transformers import pipeline

MODEL_ID = "mehtayash12345678/mlops-ag_news_classification-distilbert"
# TODO: Replace with id2label.json when merged into develop
LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def main():
    input_text = os.environ.get("INPUT_TEXT")
    if not input_text:
        raise ValueError("INPUT_TEXT environment variable is not set.")

    hf_token = os.environ.get("HF_TOKEN")

    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        token=hf_token,
    )

    result = classifier(input_text)[0]
    label_index = int(result["label"].replace("LABEL_", ""))
    label_name = LABELS[label_index]
    confidence = result["score"]

    print(f"Input: {input_text}")
    print(f"Predicted Label: {label_name}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
