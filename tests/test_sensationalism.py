"""Unit tests for sensationalism and clickbait scoring."""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.url_scraper import compute_sensationalism_score


class TestSensationalismScoring(unittest.TestCase):
    """Test suite for sensationalism analysis."""

    def test_neutral_ap_headline(self):
        title = "President signs infrastructure investment bill into law"
        body = "Officials gathered at the White House on Monday to sign the bipartisan agreement."
        score, keywords = compute_sensationalism_score(title, body)
        self.assertLessEqual(score, 0.15)
        self.assertEqual(len(keywords), 0)

    def test_all_caps_headline(self):
        title = "SHOCKING BOMBSHELL CONSPIRACY EXPOSED"
        body = "You will not believe what was found in the secret archives."
        score, keywords = compute_sensationalism_score(title, body)
        self.assertGreaterEqual(score, 0.40)

    def test_multiple_exclamation_marks(self):
        title = "You won't believe what happened next!!!"
        body = "Details inside."
        score, _ = compute_sensationalism_score(title, body)
        self.assertGreaterEqual(score, 0.35)

    def test_known_acronyms_excluded(self):
        # NASA, FBI, CIA, WHO, BBC, USA shouldn't trigger all-caps penalty
        title = "NASA and FBI announce joint technology conference"
        body = "The space agency and federal bureau of investigation met today."
        score, keywords = compute_sensationalism_score(title, body)
        self.assertLessEqual(score, 0.15)

    def test_empty_inputs(self):
        score, keywords = compute_sensationalism_score("", "")
        self.assertEqual(score, 0.0)
        self.assertEqual(keywords, [])

    def test_score_capped_at_one(self):
        extreme_title = "SHOCKING BOMBSHELL MIRACLE CURE ILLUMINATI COVER UP BANNED SECRET REVEALED!!!!!!"
        extreme_body = "DOCTORS STUNNED YOU WON T BELIEVE THIS DEEP STATE HOAX EXPOSED"
        score, _ = compute_sensationalism_score(extreme_title, extreme_body)
        self.assertLessEqual(score, 1.0)

    def test_buzzword_detection(self):
        title = "New miracle cure found in nature"
        body = "Big pharma is hiding this secret revealed by researchers."
        score, keywords = compute_sensationalism_score(title, body)
        self.assertIn("miracle cure", keywords)
        self.assertIn("secret revealed", keywords)


if __name__ == "__main__":
    unittest.main()
