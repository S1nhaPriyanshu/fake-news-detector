"""Model Training and Evaluation Module for Fake News Detection (Steps 3 & 4).

Trains and evaluates:
1. Logistic Regression (Tuned L2 Regularization)
2. Multinomial Naive Bayes
3. Calibrated Linear Support Vector Classifier (LinearSVC)
"""

import os
import json
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def split_dataset(
    X,
    y,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[Any, Any, Any, Any]:
    """Perform stratified 80% train / 20% test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: str,
    classes: Tuple[str, str] = ("REAL (0)", "FAKE (1)"),
) -> None:
    """Render a confusion matrix heatmap and save to image file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 5))

    group_names = ["True Negative (TN)", "False Positive (FP)", "False Negative (FN)", "True Positive (TP)"]
    group_counts = [f"{val:0,d}" for val in cm.flatten()]
    group_percentages = [f"{val:.2%}" for val in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=classes,
        yticklabels=classes,
        linewidths=1.5,
        linecolor="white",
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    plt.ylabel("Actual True Class", fontsize=11, fontweight="bold", labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Confusion matrix plot saved: {output_path}")


def train_and_evaluate(
    X_train,
    X_test,
    y_train,
    y_test,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> Dict[str, Dict[str, Any]]:
    """Train classifiers, evaluate metrics, and save checkpoints."""
    os.makedirs(model_dir, exist_ok=True)

    algorithms = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=2.0,
            solver="liblinear",
            random_state=42,
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear Support Vector Classifier": CalibratedClassifierCV(
            LinearSVC(C=1.0, random_state=42, dual=False),
            cv=3,
        ),
    }

    results: Dict[str, Dict[str, Any]] = {}

    for name, clf in algorithms.items():
        print(f"\n--- Training {name} ---")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="binary")
        rec = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")
        cm = confusion_matrix(y_test, y_pred)
        rep = classification_report(y_test, y_pred, target_names=["REAL", "FAKE"], output_dict=True)

        slug = name.lower().replace(" ", "_")
        model_save_path = os.path.join(model_dir, f"{slug}.joblib")
        cm_plot_path = os.path.join(model_dir, f"confusion_matrix_{slug}.png")

        # Save model
        joblib.dump(clf, model_save_path)

        # Plot and save confusion matrix
        plot_and_save_confusion_matrix(cm, name, cm_plot_path)

        # Record metrics
        results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
            "model_path": model_save_path,
            "plot_path": cm_plot_path,
            "classification_report": rep,
        }

        print(f"Results for {name}:")
        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}%")
        print(f"  Recall:    {rec * 100:.2f}%")
        print(f"  F1 Score:  {f1 * 100:.2f}%")
        print(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    # Export metrics json
    metrics_path = os.path.join(model_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll metrics successfully saved to: {metrics_path}")

    return results


def load_model(name: str = "Logistic Regression", model_dir: str = DEFAULT_MODEL_DIR):
    """Load a trained model by name from disk."""
    slug = name.lower().replace(" ", "_")
    target_path = os.path.join(model_dir, f"{slug}.joblib")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Model file '{target_path}' not found. Please run training first.")
    return joblib.load(target_path)
