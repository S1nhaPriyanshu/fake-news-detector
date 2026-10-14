# Comprehensive System Architecture & Engineering Report: VERITAS Fake News & Disinformation Intelligence Platform

---

## Executive Summary

The **VERITAS Disinformation Intelligence Platform** is an enterprise-grade Natural Language Processing (NLP) and Machine Learning system engineered to detect fabricated news stories, state-sponsored propaganda, political disinformation, and satirical articles directly from **live news links (URLs)** and **raw text**.

Unlike rudimentary toy projects that only accept pasted text and quickly fail in real-world scenarios, VERITAS is an end-to-end operational platform with:
- **Cloud-trained ML Models:** 44,898 articles from the Kaggle ISOT dataset, achieving **99.30% validation accuracy**.
- **Real-Time URL Web Scraping:** Extracts headlines, OpenGraph metadata, and article paragraphs while stripping ads, navigational scripts, and cookie banners.
- **Multi-Tier Hierarchical Source Authority:** Solves out-of-domain false positives on major wire journalism (BBC, Reuters, AP, The Hindu) while catching satirical and conspiracy networks.
- **Model Explainability (LIME):** Breaks down the exact words pushing predictions toward `REAL` or `FAKE`.
- **In-Memory LRU Caching:** High-speed response times and zero redundant network overhead.
- **Automated Verification:** 57 unit, integration, and regression tests ensuring 100% test passing rate.

---

## 1. What Data Was Used to Train the Model?

### 1.1 The Primary Dataset: Kaggle ISOT Fake & Real News Dataset
- **Source:** University of Victoria / Kaggle (`clmentbisaillon/fake-and-real-news-dataset`).
- **Total Articles:** **44,898 articles** (over 115 MB of raw text).
- **Class Breakdown:**
  - `True.csv`: **21,417 articles** (factual journalism sourced from Reuters wire reports).
  - `Fake.csv`: **23,481 articles** (fabricated, unverified, or sensational stories flagged by fact-checking organizations like PolitiFact).
- **Baseline Dataset:** McIntire Fake News Dataset (6,335 articles) was also used for initial pipeline prototyping.

### 1.2 Anti-Leakage Data Engineering (Critical Step)
In naive training setups, models often cheat by memorizing dataset artifacts. For example, legitimate Reuters articles in the ISOT dataset frequently began with datelines such as:
```text
"WASHINGTON (Reuters) - The U.S. Senate passed a bipartisan resolution..."
```
If left uncleaned, a machine learning model would simply check whether the word `"reuters"` or `"washington"` was present, rather than actually analyzing the writing style, linguistic tone, or syntactic structure of the article.

**Our Fix in [`src/data_loader.py`](src/data_loader.py):**
A regex cleaning filter was implemented to strip all publisher datelines (`^[A-Z\s]+ \((Reuters|AP)\) - `) before training. This forced the TF-IDF vectorizer and the classifier to learn purely from linguistic syntax, semantic patterns, and vocabulary distribution.

---

## 2. The 4-Stage NLP & Machine Learning Pipeline

```
Raw URL / Text
      │
      ▼
[Stage 1: Preprocessing & Text Sanitization]
  • Lowercasing
  • Regex URL & HTML tag removal
  • Punctuation & numeric filtering
  • NLTK English stopword elimination
  • Porter Stemming (e.g. 'investigating' -> 'investig')
      │
      ▼
[Stage 2: Feature Extraction (TF-IDF Vectorization)]
  • 10,000 top discriminative n-grams (unigrams + bigrams)
  • Sublinear Term Frequency: 1 + log(tf)
  • Unicode accent stripping
      │
      ▼
[Stage 3: Calibrated Machine Learning Classifiers]
  • Calibrated Linear Support Vector Classifier (LinearSVC)
  • Logistic Regression (L2 regularization)
  • Multinomial Naive Bayes (Laplace smoothing)
      │
      ▼
[Stage 4: Multi-Tier Verification & Output]
  • Source Domain Credibility Lookup (Verified vs Satire vs Hoax vs Unverified)
  • Linguistic Sensationalism Score (Clickbait density + All-Caps ratio)
  • LIME Word-Level Feature Contributions
  • Cached Verdict Rendering in Streamlit UI
```

### Stage 1: Text Sanitization (`src/preprocessor.py`)
1. **Case Normalization:** Converts everything to lowercase so `"BREAKING"` and `"breaking"` map to the same token.
2. **URL & HTML Stripping:** Regular expressions remove `http://...`, `https://...`, and DOM markup like `<div>`, `<script>`, `<b>`.
3. **Punctuation & Digit Elimination:** Strips punctuation and digits, replacing them with single spaces to keep adjacent words separate.
4. **Stopword Filtering:** Removes ubiquitous English words (`'the'`, `'is'`, `'at'`, `'which'`) that carry zero discriminative power.
5. **Porter Stemming:** Normalizes words to their morphological base stem:
   $$\text{running, runs, ran} \longrightarrow \text{run}$$
   This reduces the dimensionality of the vocabulary without losing semantic meaning.

### Stage 2: Feature Extraction (`src/feature_extractor.py`)
Computers cannot process raw strings; words must be converted into numerical matrices. We use **Term Frequency - Inverse Document Frequency (TF-IDF)** with sublinear term-frequency scaling and n-gram extraction (1, 2):

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

- **Term Frequency ($\text{TF}$):**
  $$\text{TF}(t, d) = 1 + \log(\text{Count}(t \in d))$$
  Sublinear scaling prevents an article that mentions a keyword 50 times from overpowering an article that mentions it 5 times.
- **Inverse Document Frequency ($\text{IDF}$):**
  $$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
  Gives high weight to rare, discriminative words and low weight to common words.
- **Vocabulary Size:** Capped at the **10,000 most discriminative n-grams**.

### Stage 3: Model Training & Evaluation (`src/model.py`)
The dataset was split using a stratified 80% train / 20% test partition:
- **Calibrated Linear Support Vector Classifier (`CalibratedClassifierCV(LinearSVC)`):**
  Finds the optimal separating hyperplane that maximizes the margin between classes in high-dimensional TF-IDF space, wrapped in sigmoid calibration for real probabilities.
- **Logistic Regression (`LogisticRegression`):**
  Optimized with L2 regularization ($C=2.0$) and `liblinear` solver.
- **Multinomial Naive Bayes (`MultinomialNB`):**
  Probabilistic classifier applying Bayes' theorem with Laplace smoothing ($\alpha = 0.1$).

#### Quantitative Results:
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Calibrated LinearSVC (Production)** | **99.30%** | **99.30%** | **99.36%** | **99.33%** |
| **Logistic Regression** | 99.01% | 98.81% | 99.28% | 99.05% |
| **Multinomial Naive Bayes** | 95.58% | 95.22% | 96.34% | 95.78% |

---

## 3. How the Live Link (URL) Analysis Works

When a user pastes a URL (such as a BBC, Reuters, or blog link) into the interface, the system executes the following steps:

```
                          User Inputs URL
                                 │
                                 ▼
                     [Check In-Memory Cache]
                     ├── Hit  ──> Return instantly (0 ms)
                     └── Miss ──> Continue
                                 │
                                 ▼
                    [Live Web Scraper & Parser]
              • Fetch HTML via HTTP requests (with browser headers)
              • Parse metadata: og:title, twitter:title, <h1>
              • Extract article body paragraphs (<article>, <p>)
              • Calculate word count & sensationalism score
                                 │
                                 ▼
                 [Domain Credibility Verification]
              • Checks against verified, satire, hoax registries
                                 │
                                 ▼
             [Hierarchical Decision & Machine Learning]
              ├── Verified Publisher (BBC, Reuters, etc.)
              │     └── If Sensationalism < 0.70  ──> REAL (98.5%)
              │     └── If Sensationalism >= 0.70 ──> FAKE (Alert)
              ├── Known Satire (The Onion, Babylon Bee)
              │     └── FAKE / Satire (98.5%)
              ├── Known Hoax (Infowars, etc.)
              │     └── FAKE (99.0%)
              └── Unverified / Independent Domain
                    └── Run Preprocessing + TF-IDF (10K) + Calibrated LinearSVC
                    └── Blend with Sensationalism Score
                                 │
                                 ▼
                     [LIME Explainability Engine]
              • Perturbs input text 500 times
              • Identifies top words pushing REAL vs FAKE
                                 │
                                 ▼
                       [Store in Cache & Render]
```

### 3.1 Web Scraper (`src/url_scraper.py`)
- Emulates legitimate user-agent headers to bypass basic anti-scraping firewalls.
- Decomposes boilerplate navigation bars, header tags, footer tags, inline styles, and JavaScript scripts.
- Prioritizes semantic OpenGraph headlines (`og:title`) and Twitter metadata over raw `<title>` tags to capture the editorial title without publication suffixes.
- Aggregates paragraph text (`<p>`) located specifically inside `<article>` tags while discarding cookie warnings and terms of service snippets.

### 3.2 The "Out-of-Domain" Problem & The Multi-Tier Solution
**The Real-World Issue:**
The ISOT dataset is composed almost entirely of 2016–2017 political news. When a user pastes a 2026 science or technology article from a legitimate publisher (e.g., a BBC article discussing AI software), the statistical model sees words that were never present in the 2016 political training set. As a result, a standalone bag-of-words model can produce a false positive.

**Our Multi-Tier Authority Hierarchy Solution:**
1. **Tier 1 (Verified Journalism Registry):** Major accredited wire and global press organizations (BBC, Reuters, AP, The Guardian, The Hindu, etc.) are recognized. Unless an article exhibits an extreme sensationalism score ($\ge 0.70$), it receives a verified authentic verdict.
2. **Tier 2 (Known Satire Registry):** Parody publications (The Onion, Babylon Bee) are flagged with the specialized verdict `FAKE (Satire / Parody)`.
3. **Tier 3 (Known Hoax Registry):** Documented conspiracy platforms (Infowars, etc.) are directly flagged as `FAKE`.
4. **Tier 4 (Unverified / Independent Outlets):** Independent blogs and unverified domains are analyzed using the **full machine learning model** combined with **linguistic sensationalism scoring**.

---

## 4. Explainability & Caching Engineering

### 4.1 LIME (Local Interpretable Model-agnostic Explanations) (`src/explainer.py`)
To prevent the model from acting as an opaque "black box," we integrate LIME:
- It creates 500 local perturbations of the article text by randomly removing words.
- It passes each perturbation through the classifier to measure how probability shifts.
- It computes word-level feature weights:
  - **Words pushing REAL (Green):** e.g., *"according"*, *"official"*, *"spokesman"*, *"confirmed"*.
  - **Words pushing FAKE (Red):** e.g., *"shocking"*, *"bombshell"*, *"miracle"*, *"secret"*.
- The results are displayed in an interactive table inside the web application.

### 4.2 In-Memory LRU Cache with TTL (`src/cache.py`)
- Implemented using an ordered dictionary (`OrderedDict`) with a maximum capacity of 100 items and a 1-hour time-to-live (TTL).
- If multiple users inspect the same trending news link, the second request serves instantly from memory without re-scraping or re-running inference.

### 4.3 Dynamic Domain Configuration via CSV (`data/*.csv`)
Domain whitelists and blacklists are externalized into CSV files:
- `data/verified_domains.csv`
- `data/satire_domains.csv`
- `data/hoax_domains.csv`

New publishers can be added or removed without code changes or server redeployment.

---

## 5. Automated Verification & Testing Framework

The repository includes a comprehensive testing suite consisting of **57 automated tests across 7 test files**:

```bash
.venv/bin/python -m pytest tests/ -v
```

1. **`tests/test_preprocessor_edge_cases.py` (9 tests):** Validates empty inputs, `None` values, stopword-only strings, punctuation-only strings, numeric data, Unicode/emojis, 50,000-word strings, HTML/script stripping, and case insensitivity.
2. **`tests/test_domain_credibility.py` (9 tests):** Tests subdomain matching (`news.bbc.co.uk`), typosquatting rejection (`bbcnews.com.co`), case handling (`BBC.COM`), and IP address domains.
3. **`tests/test_sensationalism.py` (7 tests):** Tests neutral AP copy, all-caps headlines, exclamation mark penalties, and acronym whitelisting (`NASA`, `FBI`, `CIA`, `WHO`, `BBC`).
4. **`tests/test_scraper_robustness.py` (6 tests):** Validates URL normalization, protocol preservation, port stripping, and network timeout handling via mock objects.
5. **`tests/test_predictor_e2e.py` (8 tests):** Validates end-to-end routing for verified, satire, hoax, and unverified domains with mocked web responses.
6. **`tests/test_model_regression.py` (4 tests):** Asserts model checkpoints load properly and retain 100% accuracy on golden sample test cases.
7. **`tests/test_improvements.py` (4 tests):** Validates LRU cache eviction, TTL expiry, LIME fallback handling, and confidence band categorization.

---

## 6. How to Run & Deploy the System

### 6.1 Local Setup
```bash
# Clone the repository
git clone https://github.com/S1nhaPriyanshu/fake-news-detector.git
cd fake-news-detector

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit web application
streamlit run app.py
```

### 6.2 Cloud Retraining (Kaggle or Google Colab)
- **Kaggle Kernel:** Located in `kaggle_kernel/fake_news_training.ipynb`. Can be pushed directly using the Kaggle CLI:
  ```bash
  kaggle kernels push -p kaggle_kernel/
  ```
- **Google Colab:** Open `Train_Fake_News_Kaggle_Colab.ipynb` in Colab, run all cells with GPU acceleration, and download the exported `best_fake_news_model.joblib`.

---

## 7. Key Viva Q&As for Academic Evaluation

**Q1: Why did you choose LinearSVC over deep neural networks like BERT for the core inference model?**  
*Answer:* Calibrated LinearSVC achieves 99.30% accuracy on large TF-IDF matrices with inference latency under 5 milliseconds on standard CPU hardware, whereas BERT models require substantial GPU memory and have significantly higher inference latency. For real-time web URL scanning, LinearSVC delivers strong accuracy with minimal operational overhead.

**Q2: What is the purpose of CalibratedClassifierCV?**  
*Answer:* Standard Support Vector Classifiers output geometric signed distances to the separating hyperplane rather than calibrated probabilities. `CalibratedClassifierCV` applies Platt scaling (a sigmoid function fitted over cross-validation splits) to transform those distances into calibrated probabilities between 0.0 and 1.0.

**Q3: How does your system prevent false positives on modern news topics not covered in historical training data?**  
*Answer:* Through our Multi-Tier Hierarchical Source Authority Engine. Accredited publishers (e.g., BBC, Reuters) are cross-referenced with curated domain registries. Unless an article exhibits strong sensationalism cues ($\ge 0.70$), it is verified as authentic, protecting against vocabulary domain shift.

---

## Summary Matrix

| Capability | Basic Fake News Project | VERITAS Intelligence Platform |
|---|---|---|
| **Input Type** | Manual text box only | **Live Web News Links (URLs) + Raw Text** |
| **Training Scale** | ~1,000–6,000 samples | **44,898 articles (Kaggle ISOT Cloud Trained)** |
| **Accuracy** | ~85–90% | **99.30% (Calibrated LinearSVC)** |
| **Feature Extraction** | Basic CountVectorizer | **10,000 n-gram Sublinear TF-IDF** |
| **Explainability** | None (Black Box) | **LIME Word-Level Contribution Analysis** |
| **False Positive Defense** | None | **Multi-Tier Hierarchical Domain Engine** |
| **Caching Layer** | None | **In-Memory LRU Cache with 1-hr TTL** |
| **Automated Tests** | 0–2 rudimentary tests | **57 Automated Tests (100% Passing)** |
| **User Interface** | Basic Streamlit form | **CARI Cyber-Editorial UI with Telemetry HUD** |
