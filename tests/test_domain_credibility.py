"""Unit tests for domain credibility evaluation in the fake news detector."""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.url_scraper import (
    evaluate_domain_credibility,
    VERIFIED_NEWS_DOMAINS,
    KNOWN_SATIRE_DOMAINS,
    KNOWN_HOAX_DOMAINS,
)


class TestDomainCredibilityEdgeCases(unittest.TestCase):
    """Test suite for domain credibility checks."""

    def test_subdomain_matching(self):
        res = evaluate_domain_credibility("news.bbc.co.uk")
        self.assertEqual(res["status"], "VERIFIED_OUTLET")

    def test_typosquatting_rejection(self):
        # A typosquatted domain that ends with .com.co or similar prefix
        res = evaluate_domain_credibility("bbcnews.com.co")
        self.assertEqual(res["status"], "UNVERIFIED_DOMAIN")

    def test_case_insensitivity(self):
        res = evaluate_domain_credibility("BBC.COM")
        self.assertEqual(res["status"], "VERIFIED_OUTLET")
        res_onion = evaluate_domain_credibility("TheOnion.COM")
        self.assertEqual(res_onion["status"], "KNOWN_SATIRE")

    def test_empty_domain(self):
        res = evaluate_domain_credibility("")
        self.assertEqual(res["status"], "UNVERIFIED_DOMAIN")

    def test_ip_address_domain(self):
        res = evaluate_domain_credibility("192.168.1.1")
        self.assertEqual(res["status"], "UNVERIFIED_DOMAIN")

    def test_all_verified_domains_present(self):
        for domain in VERIFIED_NEWS_DOMAINS:
            res = evaluate_domain_credibility(domain)
            self.assertEqual(res["status"], "VERIFIED_OUTLET", f"Failed on verified domain: {domain}")

    def test_all_satire_domains_present(self):
        for domain in KNOWN_SATIRE_DOMAINS:
            res = evaluate_domain_credibility(domain)
            self.assertEqual(res["status"], "KNOWN_SATIRE", f"Failed on satire domain: {domain}")

    def test_all_hoax_domains_present(self):
        for domain in KNOWN_HOAX_DOMAINS:
            res = evaluate_domain_credibility(domain)
            self.assertEqual(res["status"], "KNOWN_HOAX", f"Failed on hoax domain: {domain}")

    def test_random_domain_unverified(self):
        res = evaluate_domain_credibility("somecrazydomain123.xyz")
        self.assertEqual(res["status"], "UNVERIFIED_DOMAIN")


if __name__ == "__main__":
    unittest.main()
