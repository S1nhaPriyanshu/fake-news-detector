"""URL Scraper and Article Content Extractor for Fake News Detection.

Fetches live web news articles from any valid URL, extracts headlines and body text,
evaluates domain credibility against known publisher databases, and calculates
sensationalism / clickbait indicators.
"""

import re
import urllib.request
from urllib.parse import urlparse
from typing import Dict, Any, List, Set, Tuple
from bs4 import BeautifulSoup

import os
import csv

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _load_domains_from_csv(filename: str, fallback_set: Set[str]) -> Set[str]:
    """Load domains from CSV if file exists; otherwise fallback to hardcoded set."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return set(fallback_set)
    try:
        loaded = set()
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and row[0].strip():
                    loaded.add(row[0].strip().lower())
        return loaded if loaded else set(fallback_set)
    except Exception:
        return set(fallback_set)


# Curated global verified journalism, science, and news outlets
_DEFAULT_VERIFIED_NEWS_DOMAINS: Set[str] = {
    # Major Global & Wire Agencies
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "afp.com", "bloomberg.com",
    # Prestigious International Press
    "theguardian.com", "nytimes.com", "washingtonpost.com", "wsj.com", "ft.com",
    "theatlantic.com", "economist.com", "politico.com", "time.com", "forbes.com",
    "aljazeera.com", "dw.com", "france24.com", "euronews.com", "independent.co.uk",
    "telegraph.co.uk", "news.sky.com", "npr.org", "pbs.org", "cbsnews.com",
    "nbcnews.com", "abcnews.go.com", "cnn.com", "cnbc.com", "usatoday.com",
    # Leading Indian National Press
    "thehindu.com", "indianexpress.com", "hindustantimes.com", "ndtv.com",
    "indiatoday.in", "timesofindia.indiatimes.com", "economictimes.indiatimes.com",
    "livemint.com", "theprint.in", "scroll.in", "thewire.in", "business-standard.com",
    # Science, Technology & Research Journals
    "nature.com", "sciencemag.org", "sciencedaily.com", "scientificamerican.com",
    "newscientist.com", "technologyreview.com", "wired.com", "arstechnica.com",
    "theverge.com", "techcrunch.com", "zdnet.com", "phys.org", "space.com",
    "nationalgeographic.com",
}

# Known Humorous Satire / Parody Sites (Not factual journalism)
_DEFAULT_KNOWN_SATIRE_DOMAINS: Set[str] = {
    "theonion.com", "babylonbee.com", "clickhole.com", "newsthump.com",
    "thepoke.co.uk", "waterfordwhispersnews.com", "thebeaverton.com",
    "duffelblog.com", "borowitzreport.com", "satirewire.com",
}

# Documented Disinformation & Conspiracy Sites
_DEFAULT_KNOWN_HOAX_DOMAINS: Set[str] = {
    "infowars.com", "worldnewsdailyreport.com", "beforeitsnews.com",
    "naturalnews.com", "nationalreport.net", "newspunch.com", "yournewswire.com",
    "conservativetribune.com", "wnd.com", "breitbart.com", "veteranstoday.com",
    "thegatewaypundit.com", "activistpost.com", "realrawnews.com",
}

VERIFIED_NEWS_DOMAINS: Set[str] = _load_domains_from_csv("verified_domains.csv", _DEFAULT_VERIFIED_NEWS_DOMAINS)
KNOWN_SATIRE_DOMAINS: Set[str] = _load_domains_from_csv("satire_domains.csv", _DEFAULT_KNOWN_SATIRE_DOMAINS)
KNOWN_HOAX_DOMAINS: Set[str] = _load_domains_from_csv("hoax_domains.csv", _DEFAULT_KNOWN_HOAX_DOMAINS)

SENSATIONAL_KEYWORDS: List[str] = [
    "shocking", "miracle cure", "miracle", "banned", "secret revealed",
    "they don t want you to know", "doctors stunned", "doctors in shock",
    "unbelievable", "bombshell", "conspiracy", "illuminati", "cover up",
    "mind control", "obliterates", "alien pyramid", "synthetic blood",
    "hoax exposed", "you won t believe", "share before deleted",
    "breaking bombshell", "deep state", "hollow earth", "microchip vaccine",
]


def normalize_url(url: str) -> str:
    """Ensure URL has a valid HTTP/HTTPS scheme."""
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    return url


def extract_domain(url: str) -> str:
    """Extract clean base domain name from a URL."""
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc.split(":")[0]
    except Exception:
        return ""


def evaluate_domain_credibility(domain: str) -> Dict[str, Any]:
    """Check domain against curated journalism, satire, and hoax directories."""
    domain = domain.lower()

    for known in VERIFIED_NEWS_DOMAINS:
        if domain == known or domain.endswith("." + known):
            return {
                "status": "VERIFIED_OUTLET",
                "label": "Verified Major News Outlet",
                "badge": "🟢 Verified Publisher",
                "description": f"Domain '{domain}' is an established, accredited journalism organization with editorial review standards.",
            }

    for known in KNOWN_SATIRE_DOMAINS:
        if domain == known or domain.endswith("." + known):
            return {
                "status": "KNOWN_SATIRE",
                "label": "Known Satire / Parody Publisher",
                "badge": "⚠️ Satire Publication",
                "description": f"Domain '{domain}' is a documented satirical publication; articles are comedic, not factual.",
            }

    for known in KNOWN_HOAX_DOMAINS:
        if domain == known or domain.endswith("." + known):
            return {
                "status": "KNOWN_HOAX",
                "label": "Documented Misinformation Source",
                "badge": "🚨 Known Disinformation",
                "description": f"Domain '{domain}' has a documented record of publishing conspiracy theories or fabricated stories.",
            }

    return {
        "status": "UNVERIFIED_DOMAIN",
        "label": "Independent / Unverified Source",
        "badge": "⚪ Unverified Publisher",
        "description": f"Domain '{domain}' is not in the whitelist of major international news agencies.",
    }


def compute_sensationalism_score(title: str, text: str) -> Tuple[float, List[str]]:
    """Compute linguistic sensationalism and clickbait density score."""
    combined = (title + " " + text).lower()
    found_buzzwords = []

    for phrase in SENSATIONAL_KEYWORDS:
        if phrase in combined:
            found_buzzwords.append(phrase)

    exclamation_count = title.count("!")
    exclamation_penalty = min(0.35, exclamation_count * 0.18)

    words = title.split()
    caps_words = [w for w in words if len(w) >= 3 and w.isupper() and w not in {"NASA", "WHO", "FBI", "CIA", "USA", "COVID", "BBC", "AI"}]
    caps_penalty = min(0.35, len(caps_words) * 0.15)

    keyword_penalty = min(0.40, len(found_buzzwords) * 0.20)

    total_score = min(1.0, exclamation_penalty + caps_penalty + keyword_penalty)
    return round(total_score, 3), found_buzzwords


def extract_article_from_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Fetch live web news article from URL and extract headline and body."""
    norm_url = normalize_url(url)
    domain = extract_domain(norm_url)
    credibility = evaluate_domain_credibility(domain)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        req = urllib.request.Request(norm_url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            html_bytes = response.read()

        soup = BeautifulSoup(html_bytes, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
            tag.decompose()

        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()

        if not title:
            tw_title = soup.find("meta", attrs={"name": "twitter:title"})
            if tw_title and tw_title.get("content"):
                title = tw_title["content"].strip()

        if not title:
            h1 = soup.find("h1")
            if h1 and h1.text:
                title = h1.text.strip()

        if not title:
            title_tag = soup.find("title")
            if title_tag and title_tag.text:
                title = title_tag.text.strip()

        clean_title = re.sub(r"\s*[-|–—]\s*(BBC News|Reuters|CNN|The New York Times|The Guardian|The Hindu|NDTV).*$", "", title, flags=re.I).strip()
        if clean_title:
            title = clean_title

        paragraphs = []
        article_elem = soup.find("article")
        search_root = article_elem if article_elem else soup

        for p in search_root.find_all("p"):
            p_text = p.get_text(separator=" ", strip=True)
            if len(p_text) > 35 and not any(skip in p_text.lower() for skip in ["cookie policy", "rights reserved", "terms of use", "privacy policy"]):
                paragraphs.append(p_text)

        full_body = " ".join(paragraphs).strip()

        if len(full_body) < 100:
            full_body = " ".join([elem.get_text(strip=True) for elem in soup.find_all(["div", "section"]) if len(elem.get_text(strip=True)) > 60])

        word_count = len(full_body.split())
        sens_score, buzzwords = compute_sensationalism_score(title, full_body)

        return {
            "success": True,
            "url": norm_url,
            "domain": domain,
            "title": title or "News Article",
            "content": full_body,
            "word_count": word_count,
            "credibility": credibility,
            "sensationalism_score": sens_score,
            "sensational_keywords_found": buzzwords,
            "error_message": None,
        }

    except Exception as exc:
        return {
            "success": False,
            "url": norm_url,
            "domain": domain,
            "title": "",
            "content": "",
            "word_count": 0,
            "credibility": credibility,
            "sensationalism_score": 0.0,
            "sensational_keywords_found": [],
            "error_message": str(exc),
        }
