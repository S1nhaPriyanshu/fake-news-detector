"""Data Preprocessing Module for Fake News Detection (Step 1 of the Pipeline).

This module handles:
1. Case normalization (converting text to lowercase).
2. Punctuation, URL, and special character removal.
3. Stopword removal (filtering out words like 'the', 'is', 'at' that add no discriminative value).
4. Stemming (reducing words to their root form using Porter Stemmer, e.g., 'running' -> 'run').
"""

import re
import string
from typing import Dict, List, Optional, Set, Union
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK data dependencies are loaded safely
try:
    _STOPWORDS: Set[str] = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOPWORDS = set(stopwords.words("english"))

_STEMMER = PorterStemmer()


def clean_raw_text(text: str) -> str:
    """Normalize raw text by lowercasing, stripping URLs, HTML, and punctuation.

    Args:
        text: Input string (headline or article body).

    Returns:
        Cleaned text string containing only lowercase alphabetic words and single spaces.
    """
    if not isinstance(text, str):
        return ""

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Strip URLs and web links
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Strip HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 4. Remove punctuation and non-alphabetic characters
    # Replacing punctuation and digits with spaces ensures words aren't accidentally joined
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 5. Collapse multiple whitespaces into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords_and_stem(
    cleaned_text: str,
    custom_stopwords: Optional[Set[str]] = None,
) -> str:
    """Remove English stopwords and apply Porter Stemming to words.

    Args:
        cleaned_text: Sanitized text string.
        custom_stopwords: Optional custom set of stopwords. Defaults to NLTK English stopwords.

    Returns:
        Space-separated string of stemmed non-stopword tokens.
    """
    stop_words = custom_stopwords if custom_stopwords is not None else _STOPWORDS
    tokens = cleaned_text.split()

    stemmed_tokens = [
        _STEMMER.stem(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(stemmed_tokens)


def preprocess_text(text: str) -> str:
    """Full Step 1 preprocessing pipeline: cleaning, stopword removal, and stemming.

    This function represents the exact standard transformation applied both during
    training and live user inference.

    Args:
        text: Raw input headline or news article.

    Returns:
        Fully preprocessed and stemmed text string ready for vectorization.
    """
    cleaned = clean_raw_text(text)
    return remove_stopwords_and_stem(cleaned)


def inspect_preprocessing_steps(text: str) -> Dict[str, Union[str, List[str]]]:
    """Break down each stage of preprocessing for interactive UI and viva explanation.

    Args:
        text: Raw text input.

    Returns:
        Dictionary detailing the state of the text at every sub-step.
    """
    raw_preview = text
    lowercased = text.lower()
    cleaned = clean_raw_text(text)
    tokens = cleaned.split()
    non_stopwords = [w for w in tokens if w not in _STOPWORDS and len(w) > 2]
    stemmed = [_STEMMER.stem(w) for w in non_stopwords]

    return {
        "raw_text": raw_preview,
        "lowercased": lowercased,
        "cleaned_text": cleaned,
        "initial_tokens_count": len(tokens),
        "filtered_tokens": non_stopwords,
        "filtered_tokens_count": len(non_stopwords),
        "stemmed_tokens": stemmed,
        "final_preprocessed_text": " ".join(stemmed),
    }


if __name__ == "__main__":
    sample = "BREAKING: Scientists have discovered 100% cure for COVID-19 in lemon juice! Read at https://hoax.com"
    print("Sample:", sample)
    print("Preprocessed:", preprocess_text(sample))
    inspection = inspect_preprocessing_steps(sample)
    print("Inspection details:", inspection)
