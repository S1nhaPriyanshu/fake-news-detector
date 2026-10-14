"""Automated unit and integration tests for the Fake News Detection pipeline."""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessor import clean_raw_text, remove_stopwords_and_stem, preprocess_text, inspect_preprocessing_steps


class TestPreprocessingPipeline(unittest.TestCase):
    """Test suite for Step 1: Preprocessing."""

    def test_clean_raw_text_lowercasing(self):
        raw = "BREAKING NEWS ALERT"
        cleaned = clean_raw_text(raw)
        self.assertEqual(cleaned, "breaking news alert")

    def test_clean_raw_text_url_removal(self):
        raw = "Check this secret http://hoax.com/article and www.fake.org immediately!"
        cleaned = clean_raw_text(raw)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("hoax.com", cleaned)
        self.assertNotIn("fake.org", cleaned)

    def test_clean_raw_text_punctuation_removal(self):
        raw = "Doctor's 'miracle' cure?! 100% true #health @cure"
        cleaned = clean_raw_text(raw)
        self.assertEqual(cleaned, "doctor s miracle cure true health cure")

    def test_remove_stopwords_and_stemming(self):
        # 'is', 'the', 'at' should be removed; 'running' stems to 'run'
        raw = "the doctor is running at the hospital"
        cleaned = clean_raw_text(raw)
        processed = remove_stopwords_and_stem(cleaned)
        tokens = processed.split()
        self.assertNotIn("the", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("at", tokens)
        self.assertIn("run", tokens)
        self.assertIn("doctor", tokens)
        self.assertIn("hospit", tokens)

    def test_inspect_preprocessing_steps(self):
        sample = "The aliens are coming! See https://aliens.com"
        inspection = inspect_preprocessing_steps(sample)
        self.assertIn("raw_text", inspection)
        self.assertIn("lowercased", inspection)
        self.assertIn("cleaned_text", inspection)
        self.assertIn("filtered_tokens", inspection)
        self.assertIn("stemmed_tokens", inspection)
        self.assertIn("final_preprocessed_text", inspection)


class TestInferencePipeline(unittest.TestCase):
    """Test suite for end-to-end inference if models have been trained."""

    def test_prediction_output_structure(self):
        try:
            from src.predictor import FakeNewsClassifier
            classifier = FakeNewsClassifier()
            result = classifier.predict("NASA announces new lunar exploration mission with international partners.")
            self.assertIn("label", result)
            self.assertIn("confidence", result)
            self.assertIn("probabilities", result)
            self.assertIn(result["label"], ["REAL", "FAKE"])
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)
        except FileNotFoundError:
            # Models not yet trained; skip
            self.skipTest("Models not yet trained, skipping inference test.")


if __name__ == "__main__":
    unittest.main()

from src.url_scraper import normalize_url, extract_domain, evaluate_domain_credibility, compute_sensationalism_score

class TestUrlScraperAndCredibility(unittest.TestCase):
    """Test suite for URL extraction and credibility signals."""

    def test_normalize_url(self):
        self.assertEqual(normalize_url("bbc.com/news"), "https://bbc.com/news")
        self.assertEqual(normalize_url("http://hoax.org"), "http://hoax.org")

    def test_extract_domain(self):
        self.assertEqual(extract_domain("https://www.bbc.com/news/123"), "bbc.com")
        self.assertEqual(extract_domain("http://babylonbee.com/article"), "babylonbee.com")

    def test_domain_credibility(self):
        bbc_eval = evaluate_domain_credibility("bbc.com")
        self.assertEqual(bbc_eval["status"], "VERIFIED_OUTLET")
        satire_eval = evaluate_domain_credibility("theonion.com")
        self.assertEqual(satire_eval["status"], "KNOWN_SATIRE")

    def test_sensationalism_score(self):
        score, buzzwords = compute_sensationalism_score("SHOCKING MIRACLE CURE!", "Big Pharma doesn't want you to know.")
        self.assertGreater(score, 0.2)
        self.assertIn("miracle cure", buzzwords)
