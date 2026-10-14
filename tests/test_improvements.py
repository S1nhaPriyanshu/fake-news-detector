"""Unit tests verifying Phase 2 improvements: Caching, Confidence Bands, Dynamic Domains, and Explainer."""

import unittest
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.cache import PredictionCache
from src.explainer import PredictionExplainer
from src.predictor import FakeNewsClassifier
from src.url_scraper import evaluate_domain_credibility


class TestPhase2Improvements(unittest.TestCase):
    """Test suite for improvements implemented in Phase 2."""

    def test_lru_cache_operations(self):
        cache = PredictionCache(max_size=2, ttl_seconds=2)
        cache.set("https://site.com/1", {"label": "REAL"})
        cache.set("https://site.com/2", {"label": "FAKE"})

        self.assertEqual(len(cache), 2)
        hit = cache.get("https://site.com/1")
        self.assertIsNotNone(hit)
        self.assertTrue(hit.get("cached"))

        # Add 3rd item -> item 2 should be evicted because item 1 was accessed
        cache.set("https://site.com/3", {"label": "REAL"})
        self.assertIsNotNone(cache.get("https://site.com/1"))
        self.assertIsNone(cache.get("https://site.com/2"))

    def test_explainer_initialization_and_output(self):
        explainer = PredictionExplainer()
        self.assertEqual(explainer.class_names, ["REAL", "FAKE"])

        # Test on short text fallback
        res = explainer.explain_instance("hi", lambda x: None)
        self.assertFalse(res["available"])

    def test_confidence_bands_and_source(self):
        classifier = FakeNewsClassifier()
        res = classifier.predict("President signs government budget bill in Washington.")
        self.assertIn("confidence_band", res)
        self.assertIn("confidence_source", res)
        self.assertIn(res["confidence_band"], ["HIGH", "MODERATE", "LOW"])
        self.assertEqual(res["confidence_source"], "NLP Machine Learning Model")

    @patch("src.predictor.extract_article_from_url")
    def test_url_caching_behavior(self, mock_scrape):
        mock_scrape.return_value = {
            "success": True,
            "url": "https://custom-mocked-test.com/news",
            "domain": "custom-mocked-test.com",
            "title": "Local team wins regional football tournament",
            "content": "Players celebrated after the final whistle blew on Saturday afternoon.",
            "word_count": 10,
            "credibility": {"status": "UNVERIFIED_DOMAIN"},
            "sensationalism_score": 0.0,
            "sensational_keywords_found": [],
            "error_message": None,
        }
        classifier = FakeNewsClassifier()
        # First call: not cached
        first = classifier.predict_from_url("https://custom-mocked-test.com/news", use_cache=True)
        self.assertFalse(first.get("cached", False))

        # Second call: cached
        second = classifier.predict_from_url("https://custom-mocked-test.com/news", use_cache=True)
        self.assertTrue(second.get("cached", True))
        self.assertEqual(mock_scrape.call_count, 1)


if __name__ == "__main__":
    unittest.main()
