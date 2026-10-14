"""Unit tests covering edge cases for text preprocessing in the fake news detection pipeline."""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessor import clean_raw_text, remove_stopwords_and_stem, preprocess_text


class TestPreprocessorEdgeCases(unittest.TestCase):
    """Test suite covering edge cases for text preprocessing."""

    def test_empty_string(self):
        self.assertEqual(clean_raw_text(""), "")
        self.assertEqual(preprocess_text(""), "")

    def test_none_input(self):
        self.assertEqual(clean_raw_text(None), "")
        self.assertEqual(preprocess_text(None), "")

    def test_only_stopwords(self):
        raw = "the is at on for an a by with"
        self.assertEqual(preprocess_text(raw), "")

    def test_only_punctuation(self):
        raw = "!@#$%^&*()_+{}[]:;\"'<>,.?/~`|\\"
        self.assertEqual(clean_raw_text(raw), "")
        self.assertEqual(preprocess_text(raw), "")

    def test_only_numbers(self):
        raw = "12345 67890 2024"
        self.assertEqual(clean_raw_text(raw), "")
        self.assertEqual(preprocess_text(raw), "")

    def test_unicode_and_emoji(self):
        raw = "Breaking 🔥 news café résumé alert! 🚨"
        cleaned = clean_raw_text(raw)
        self.assertIn("breaking", cleaned)
        self.assertIn("news", cleaned)
        self.assertNotIn("🔥", cleaned)
        self.assertNotIn("🚨", cleaned)

    def test_very_long_text(self):
        long_text = "report " * 10000
        res = preprocess_text(long_text)
        self.assertTrue(len(res) > 0)
        self.assertEqual(res.split()[0], "report")

    def test_html_tag_removal(self):
        raw = "<div class='main'><b>Breaking</b> news! <script>alert('xss');</script></div>"
        cleaned = clean_raw_text(raw)
        self.assertNotIn("<div", cleaned)
        self.assertNotIn("</div>", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertIn("breaking", cleaned)
        self.assertIn("news", cleaned)

    def test_case_insensitivity(self):
        self.assertEqual(preprocess_text("CORONAVIRUS PANDEMIC OUTBREAK"), preprocess_text("coronavirus pandemic outbreak"))


if __name__ == "__main__":
    unittest.main()
