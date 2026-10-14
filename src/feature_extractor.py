"""Feature Extraction Module for Fake News Detection (Step 2 of the Pipeline).

Converts preprocessed text into numerical feature matrices using
Term Frequency - Inverse Document Frequency (TF-IDF) with unigrams & bigrams.
"""

import os
from typing import Optional, Tuple, List
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DEFAULT_VECTORIZER_PATH = os.path.join(DEFAULT_MODEL_DIR, "tfidf_vectorizer.joblib")


def create_vectorizer(
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Instantiate a configured TfidfVectorizer.

    Args:
        max_features: Maximum vocabulary size (10,000 top discriminative features).
        ngram_range: Range of n-values (unigrams + bigrams).
        sublinear_tf: Apply sublinear scaling (1 + log(tf)).

    Returns:
        Configured un-fitted TfidfVectorizer.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        stop_words="english",
        strip_accents="unicode",
        lowercase=True,
    )


def save_vectorizer(
    vectorizer: TfidfVectorizer,
    path: str = DEFAULT_VECTORIZER_PATH,
) -> str:
    """Serialize fitted vectorizer to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"TF-IDF vectorizer saved to: {path}")
    return path


def load_vectorizer(path: str = DEFAULT_VECTORIZER_PATH) -> TfidfVectorizer:
    """Load serialized vectorizer from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trained vectorizer not found at '{path}'. "
            "Please run 'train_pipeline.py' first."
        )
    return joblib.load(path)


def get_top_features_for_text(
    vectorizer: TfidfVectorizer,
    text: str,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """Extract top N words with highest TF-IDF scores for a specific text sample."""
    feature_names = vectorizer.get_feature_names_out()
    tfidf_matrix = vectorizer.transform([text])
    feature_index = tfidf_matrix.nonzero()[1]
    tfidf_scores = zip(
        [feature_names[i] for i in feature_index],
        [tfidf_matrix[0, x] for x in feature_index],
    )
    return sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:top_n]
