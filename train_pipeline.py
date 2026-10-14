#!/usr/bin/env python3
"""End-to-End Fake News Machine Learning Training Pipeline on Full Kaggle Dataset."""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data
from src.feature_extractor import create_vectorizer, save_vectorizer
from src.model import split_dataset, train_and_evaluate


def run_pipeline() -> None:
    """Execute the full pipeline on Kaggle 45,000-article dataset."""
    start_time = time.time()
    print("=" * 75)
    print("   TRAINING FAKE NEWS DETECTION ENGINE ON FULL KAGGLE DATASET (45,000 ARTICLES)")
    print("=" * 75)

    # Step 0: Data Loading
    print("\n[STEP 0] Loading Kaggle Dataset...")
    df = load_data()
    print(f"Total samples: {len(df)}")
    print(df["label"].value_counts().to_string())

    # Step 1 & 2: Feature Extraction (TF-IDF with Stopwords and N-grams)
    print("\n[STEP 1 & 2] Extracting 10,000 TF-IDF Features (Unigrams + Bigrams)...")
    t0 = time.time()
    vectorizer = create_vectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X = vectorizer.fit_transform(df["content"])
    y = df["label_num"].values
    print(f"Feature matrix built in {time.time() - t0:.2f}s | Shape: {X.shape[0]} documents x {X.shape[1]} features")
    save_vectorizer(vectorizer)

    # Step 3: Train-Test Split (80/20)
    print("\n[STEP 3] Splitting Data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20, random_state=42)
    print(f"Training: {X_train.shape[0]} samples | Testing: {X_test.shape[0]} samples")

    # Step 4: Model Training and Evaluation
    print("\n[STEP 4] Training and Evaluating Machine Learning Classifiers...")
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Summary Table
    print("\n" + "=" * 75)
    print(f"{'ALGORITHM':<34} | {'ACCURACY':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 75)
    for name, metrics in results.items():
        print(
            f"{name:<34} | "
            f"{metrics['accuracy'] * 100:>8.2f}% | "
            f"{metrics['precision'] * 100:>8.2f}% | "
            f"{metrics['recall'] * 100:>8.2f}% | "
            f"{metrics['f1_score'] * 100:>8.2f}%"
        )
    print("=" * 75)
    print(f"Training pipeline finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    run_pipeline()
