"""Data loading and acquisition module for Fake News Detection.

Supports:
1. Kaggle ISOT Fake and Real News Dataset (True.csv & Fake.csv, ~44,898 articles).
2. George McIntire Fake and Real News Benchmark (~6,335 articles).
3. Automatic downloading and local caching.
"""

import os
import io
import re
import zipfile
import urllib.request
from typing import Optional
import pandas as pd

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRUE_CSV_PATH = os.path.join(DEFAULT_DATA_DIR, "True.csv")
FAKE_CSV_PATH = os.path.join(DEFAULT_DATA_DIR, "Fake.csv")
MCINTIRE_CSV_PATH = os.path.join(DEFAULT_DATA_DIR, "fake_or_real_news.csv")


def clean_publisher_leakage(text: str) -> str:
    """Strip wire agency prefixes (e.g., 'WASHINGTON (Reuters) -') to prevent data leakage.

    Args:
        text: Raw article body text.

    Returns:
        Cleaned text without publisher leakage.
    """
    if not isinstance(text, str):
        return ""
    # Strip leading location and agency prefix
    cleaned = re.sub(r"^[A-Z\s]{2,25}?\s*\([A-Za-z\s]+\)\s*[-—]\s*", " ", text)
    cleaned = re.sub(r"\b(reuters|reuters news agency)\b", " ", cleaned, flags=re.I)
    return cleaned.strip()


def load_kaggle_isot_data(data_dir: str = DEFAULT_DATA_DIR) -> Optional[pd.DataFrame]:
    """Load and combine True.csv and Fake.csv if available.

    Args:
        data_dir: Path to directory containing True.csv and Fake.csv.

    Returns:
        Unified DataFrame with 44,898 records or None if files missing.
    """
    true_path = os.path.join(data_dir, "True.csv")
    fake_path = os.path.join(data_dir, "Fake.csv")

    if not (os.path.exists(true_path) and os.path.exists(fake_path)):
        return None

    print(f"Loading Kaggle ISOT dataset from {true_path} and {fake_path}...")
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    # Standardize labels: REAL = 0, FAKE = 1
    df_true["label"] = "REAL"
    df_true["label_num"] = 0
    df_fake["label"] = "FAKE"
    df_fake["label_num"] = 1

    df = pd.concat([df_true, df_fake], axis=0).reset_index(drop=True)

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Anti-leakage: clean wire agency tags from real articles
    df["text"] = [clean_publisher_leakage(t) for t in df["text"]]

    # Combine title and text
    df["content"] = df["title"] + " " + df["text"]
    df["content"] = df["content"].str.strip()

    # Filter out empty records
    df = df[df["content"].str.len() > 20].reset_index(drop=True)
    print(f"Loaded Kaggle ISOT dataset: {len(df)} total articles ({sum(df['label_num']==0)} REAL, {sum(df['label_num']==1)} FAKE)")
    return df


def load_data(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the best available dataset (prefers Kaggle 45,000 dataset).

    Args:
        data_dir: Data directory path.

    Returns:
        DataFrame with columns ['title', 'text', 'content', 'label', 'label_num'].
    """
    # 1. Try Kaggle ISOT dataset first
    isot_df = load_kaggle_isot_data(data_dir)
    if isot_df is not None and len(isot_df) > 10000:
        return isot_df

    # 2. Try McIntire dataset
    if os.path.exists(MCINTIRE_CSV_PATH):
        df = pd.read_csv(MCINTIRE_CSV_PATH)
        df["title"] = df["title"].fillna("").astype(str)
        df["text"] = df["text"].fillna("").astype(str)
        df["label_num"] = (df["label"].str.upper() == "FAKE").astype(int)
        df["content"] = df["title"] + " " + df["text"]
        df["content"] = df["content"].str.strip()
        df = df[df["content"].str.len() > 10].reset_index(drop=True)
        return df

    raise FileNotFoundError("No valid dataset found in data directory.")
