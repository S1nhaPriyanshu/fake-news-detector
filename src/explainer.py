"""Model Interpretability and Explainability Module (LIME Integration).

Extracts word-level feature contributions to explain why a specific
article was classified as REAL or FAKE.
"""

from typing import List, Tuple, Dict, Any, Callable
from lime.lime_text import LimeTextExplainer
import numpy as np


class PredictionExplainer:
    """LIME Text Explainer wrapper for NLP models."""

    def __init__(self, class_names: List[str] = None):
        """Initialize LimeTextExplainer."""
        if class_names is None:
            class_names = ["REAL", "FAKE"]
        self.class_names = class_names
        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            split_expression=r"\W+",
            bow=True,
            random_state=42,
        )

    def explain_instance(
        self,
        text: str,
        predict_fn: Callable[[List[str]], np.ndarray],
        num_features: int = 8,
        num_samples: int = 500,
    ) -> Dict[str, Any]:
        """Generate local explanation for a text classification.

        Args:
            text: Raw input text.
            predict_fn: Function accepting a list of raw text strings and returning probability matrix (N, 2).
            num_features: Number of top influential words to extract.
            num_samples: Perturbation samples for LIME local surrogate model.

        Returns:
            Dictionary with feature weights, supporting words for REAL and FAKE.
        """
        if not text or len(text.strip().split()) < 3:
            return {
                "available": False,
                "reason": "Text is too short to generate local explanation.",
                "weights": [],
            }

        try:
            exp = self.explainer.explain_instance(
                text_instance=text,
                classifier_fn=predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                labels=[1],  # Explain wrt FAKE class
            )

            # Raw weights: list of (word, weight) where positive weight -> FAKE, negative weight -> REAL
            raw_weights: List[Tuple[str, float]] = exp.as_list(label=1)

            fake_contributors = []
            real_contributors = []

            for word, weight in raw_weights:
                rounded_wt = round(float(weight), 4)
                if rounded_wt > 0:
                    fake_contributors.append({"word": word, "weight": rounded_wt, "indicates": "FAKE"})
                else:
                    real_contributors.append({"word": word, "weight": abs(rounded_wt), "indicates": "REAL"})

            # Sort by absolute impact
            fake_contributors.sort(key=lambda x: x["weight"], reverse=True)
            real_contributors.sort(key=lambda x: x["weight"], reverse=True)

            return {
                "available": True,
                "weights": [{"word": w, "weight": round(float(wt), 4)} for w, wt in raw_weights],
                "fake_contributors": fake_contributors,
                "real_contributors": real_contributors,
            }
        except Exception as e:
            return {
                "available": False,
                "reason": f"Could not generate LIME explanation: {str(e)}",
                "weights": [],
            }
