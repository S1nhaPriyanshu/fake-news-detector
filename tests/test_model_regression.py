"""Regression test suite verifying model file loading and inference against known golden samples."""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.feature_extractor import load_vectorizer
from src.model import load_model
from src.predictor import FakeNewsClassifier


class TestModelRegression(unittest.TestCase):
    """Ensure trained models and vectorizer persist stable classification performance."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = FakeNewsClassifier()

    def test_vectorizer_loads_and_transforms(self):
        vec = load_vectorizer()
        self.assertTrue(hasattr(vec, "transform"))
        res = vec.transform(["President speaks at international summit"])
        self.assertEqual(res.shape[1], 10000)

    def test_models_load_successfully(self):
        for model_name in ["Logistic Regression", "Multinomial Naive Bayes", "Linear Support Vector Classifier"]:
            model = load_model(model_name)
            self.assertTrue(hasattr(model, "predict"))

    def test_golden_real_samples(self):
        real_samples = [
            "WASHINGTON (Reuters) - The U.S. Senate passed a bipartisan budget resolution on Thursday to fund federal operations.",
            "LONDON (Reuters) - British health authorities confirmed new medical guidelines following scientific consultations.",
        ]
        for sample in real_samples:
            res = self.classifier.predict(sample)
            self.assertEqual(res["label"], "REAL", f"Sample falsely classified as FAKE: {sample[:60]}")

    def test_golden_fake_samples(self):
        fake_samples = [
            "SHOCKING: Secret whistleblower exposes synthetic microchip agenda implemented through municipal drinking supplies!",
            "BREAKING BOMBSHELL: Hollywood elite underground tunnel network uncovered by independent patriots, shares banned everywhere!",
        ]
        for sample in fake_samples:
            res = self.classifier.predict(sample)
            self.assertEqual(res["label"], "FAKE", f"Sample falsely classified as REAL: {sample[:60]}")


if __name__ == "__main__":
    unittest.main()
