
import re


def clean_text(text: str) -> str:
    # Convert all letters to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Replace multiple spaces with one space
    text = re.sub(r"\s+", " ", text)

    # Remove spaces from the beginning and end
    text = text.strip()

    return text
