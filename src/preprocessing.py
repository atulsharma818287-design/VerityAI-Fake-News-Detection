import re


def clean_text(text: str) -> str:
    """Basic normalization for news text."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
