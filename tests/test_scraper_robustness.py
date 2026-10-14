"""Unit tests for URL normalization and domain extraction robustness."""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.url_scraper import (
    normalize_url,
    extract_domain,
    extract_article_from_url,
)


class TestScraperRobustness(unittest.TestCase):
    """Test suite for URL scraping handling."""

    def test_normalize_url_adds_https(self):
        self.assertEqual(normalize_url("bbc.com"), "https://bbc.com")
        self.assertEqual(normalize_url("www.theguardian.com/world"), "https://www.theguardian.com/world")

    def test_normalize_url_preserves_http(self):
        self.assertEqual(normalize_url("http://insecure-site.org"), "http://insecure-site.org")

    def test_extract_domain_strips_www(self):
        self.assertEqual(extract_domain("https://www.reuters.com/business"), "reuters.com")
        self.assertEqual(extract_domain("http://reuters.com"), "reuters.com")

    def test_extract_domain_invalid_url(self):
        self.assertEqual(extract_domain(""), "")

    def test_extract_domain_with_port(self):
        self.assertEqual(extract_domain("https://localhost:8501/dashboard"), "localhost")

    @patch("urllib.request.urlopen")
    def test_extract_article_timeout_or_error(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError("Connection timed out")
        res = extract_article_from_url("https://unreachable-domain-123.org", timeout=1)
        self.assertFalse(res["success"])
        self.assertIn("error_message", res)
        self.assertIsNotNone(res["error_message"])


if __name__ == "__main__":
    unittest.main()
