"""End-to-end integration and mock tests for FakeNewsClassifier."""

import unittest
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.predictor import FakeNewsClassifier


class TestPredictorE2E(unittest.TestCase):
    """Test suite for classifier inference across URLs and text inputs."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = FakeNewsClassifier()

    def test_empty_text_input(self):
        res = self.classifier.predict("")
        self.assertEqual(res["label"], "UNKNOWN")
        self.assertEqual(res["confidence"], 0.0)

    @patch("src.predictor.extract_article_from_url")
    def test_verified_outlet_returns_real(self, mock_scrape):
        mock_scrape.return_value = {
            "success": True,
            "url": "https://www.bbc.com/news/world-12345",
            "domain": "bbc.com",
            "title": "Global leaders meet for climate summit",
            "content": "Representatives from over one hundred countries gathered today to discuss commitments.",
            "word_count": 14,
            "credibility": {
                "status": "VERIFIED_OUTLET",
                "label": "Verified Major News Outlet",
                "badge": "🟢 Verified Publisher",
                "description": "Accredited news agency."
            },
            "sensationalism_score": 0.0,
            "sensational_keywords_found": [],
            "error_message": None,
        }
        res = self.classifier.predict_from_url("https://www.bbc.com/news/world-12345")
        self.assertEqual(res["label"], "REAL")
        self.assertGreaterEqual(res["confidence"], 0.95)

    @patch("src.predictor.extract_article_from_url")
    def test_satire_domain_returns_fake_satire(self, mock_scrape):
        mock_scrape.return_value = {
            "success": True,
            "url": "https://theonion.com/article-123",
            "domain": "theonion.com",
            "title": "Man accidentally drinks tap water, survives",
            "content": "Area man astonished as standard municipal tap water proves harmless.",
            "word_count": 10,
            "credibility": {
                "status": "KNOWN_SATIRE",
                "label": "Known Satire / Parody Publisher",
                "badge": "⚠️ Satire Publication",
                "description": "Satirical organization."
            },
            "sensationalism_score": 0.0,
            "sensational_keywords_found": [],
            "error_message": None,
        }
        res = self.classifier.predict_from_url("https://theonion.com/article-123")
        self.assertIn("FAKE", res["label"])
        self.assertIn("Satire", res["label"])

    @patch("src.predictor.extract_article_from_url")
    def test_hoax_domain_returns_fake(self, mock_scrape):
        mock_scrape.return_value = {
            "success": True,
            "url": "https://infowars.com/breaking-conspiracy",
            "domain": "infowars.com",
            "title": "Secret globalist plot exposed by whistleblowers",
            "content": "Documents allegedly leak regarding clandestine operations worldwide.",
            "word_count": 8,
            "credibility": {
                "status": "KNOWN_HOAX",
                "label": "Documented Misinformation Source",
                "badge": "🚨 Known Disinformation",
                "description": "Conspiracy platform."
            },
            "sensationalism_score": 0.35,
            "sensational_keywords_found": [],
            "error_message": None,
        }
        res = self.classifier.predict_from_url("https://infowars.com/breaking-conspiracy")
        self.assertEqual(res["label"], "FAKE")
        self.assertGreaterEqual(res["confidence"], 0.95)

    @patch("src.predictor.extract_article_from_url")
    def test_unverified_domain_uses_ml_model(self, mock_scrape):
        mock_scrape.return_value = {
            "success": True,
            "url": "https://independent-daily-blog.xyz/post/1",
            "domain": "independent-daily-blog.xyz",
            "title": "Federal Reserve updates monetary interest policy guidance",
            "content": "Central bank chair announced standard rate adjustments following economic assessment.",
            "word_count": 11,
            "credibility": {
                "status": "UNVERIFIED_DOMAIN",
                "label": "Independent / Unverified Source",
                "badge": "⚪ Unverified Publisher",
                "description": "Unverified."
            },
            "sensationalism_score": 0.0,
            "sensational_keywords_found": [],
            "error_message": None,
        }
        res = self.classifier.predict_from_url("https://independent-daily-blog.xyz/post/1")
        self.assertIn(res["label"], ["REAL", "FAKE"])
        self.assertIsNotNone(res.get("model_used"))

    @patch("src.predictor.extract_article_from_url")
    def test_scrape_failure_verified_outlet(self, mock_scrape):
        mock_scrape.return_value = {
            "success": False,
            "url": "https://reuters.com/blocked-paywall",
            "domain": "reuters.com",
            "title": "",
            "content": "",
            "word_count": 0,
            "credibility": {
                "status": "VERIFIED_OUTLET",
                "label": "Verified Major News Outlet",
                "badge": "🟢 Verified Publisher",
                "description": "Established."
            },
            "sensationalism_score": 0.0,
            "sensational_keywords_found": [],
            "error_message": "401 Unauthorized",
        }
        res = self.classifier.predict_from_url("https://reuters.com/blocked-paywall")
        self.assertEqual(res["label"], "REAL")
        self.assertFalse(res["success"])

    @patch("src.predictor.extract_article_from_url")
    def test_scrape_failure_unknown_domain(self, mock_scrape):
        mock_scrape.return_value = {
            "success": False,
            "url": "https://unknown-broken.com",
            "domain": "unknown-broken.com",
            "title": "",
            "content": "",
            "word_count": 0,
            "credibility": {
                "status": "UNVERIFIED_DOMAIN",
                "label": "Independent / Unverified Source",
                "badge": "⚪ Unverified Publisher",
                "description": "Unverified."
            },
            "sensationalism_score": 0.0,
            "sensational_keywords_found": [],
            "error_message": "404 Not Found",
        }
        res = self.classifier.predict_from_url("https://unknown-broken.com")
        self.assertEqual(res["label"], "UNKNOWN")

    def test_url_auto_detection(self):
        with patch.object(self.classifier, "predict_from_url") as mock_url_pred:
            mock_url_pred.return_value = {"mocked": True}
            self.classifier.predict("https://example.com/test-article")
            mock_url_pred.assert_called_once()


if __name__ == "__main__":
    unittest.main()
