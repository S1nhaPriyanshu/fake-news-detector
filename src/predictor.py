import warnings
warnings.filterwarnings("ignore")

import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from src.preprocessor import preprocess_text, inspect_preprocessing_steps
from src.feature_extractor import load_vectorizer, get_top_features_for_text
from src.model import load_model, DEFAULT_MODEL_DIR
from src.url_scraper import extract_article_from_url, compute_sensationalism_score
from src.cache import GLOBAL_CACHE
from src.explainer import PredictionExplainer


class FakeNewsClassifier:
    """Hybrid Classifier combining NLP Machine Learning, URL scraping, and domain credibility."""

    def __init__(self, default_model: str = "Linear Support Vector Classifier", model_dir: str = DEFAULT_MODEL_DIR):
        """Initialize classifier by loading vectorizer, models, and explainer."""
        self.model_dir = model_dir
        self.explainer = PredictionExplainer(class_names=["REAL", "FAKE"])
        self.reload_models(default_model)

    def reload_models(self, default_model: str = "Linear Support Vector Classifier") -> None:
        """Load or reload model weights from disk."""
        self.vectorizer = load_vectorizer()
        self.available_models = [
            "Linear Support Vector Classifier",
            "Logistic Regression",
            "Multinomial Naive Bayes",
        ]
        self.models = {}

        for name in self.available_models:
            try:
                self.models[name] = load_model(name, model_dir=self.model_dir)
            except Exception as e:
                print(f"Note: Model {name} could not be loaded: {e}")

        # Check for custom Colab-trained model
        colab_path = os.path.join(self.model_dir, "best_fake_news_model.joblib")
        if os.path.exists(colab_path):
            try:
                import joblib
                self.models["Colab Best Model"] = joblib.load(colab_path)
                self.available_models.insert(0, "Colab Best Model")
            except Exception as err:
                print(f"Error loading Colab model: {err}")

        if default_model in self.models:
            self.current_model_name = default_model
        elif self.models:
            self.current_model_name = list(self.models.keys())[0]
        else:
            self.current_model_name = "None"

    def _get_confidence_band(self, confidence: float) -> str:
        """Categorize confidence score into HIGH, MODERATE, or LOW."""
        if confidence >= 0.85:
            return "HIGH"
        elif confidence >= 0.65:
            return "MODERATE"
        return "LOW"

    def predict(
        self,
        text: str,
        model_name: Optional[str] = None,
        explain: bool = False,
    ) -> Dict[str, Any]:
        """Classify input text or headline with confidence, calibration, and explainability.

        Automatically detects if input is a web URL and routes to URL analysis.
        """
        stripped = text.strip() if text else ""

        # Auto-detect if user entered a URL into the text box
        if (stripped.startswith("http://") or stripped.startswith("https://") or
            (len(stripped.split()) == 1 and any(ext in stripped for ext in [".com", ".org", ".net", ".in", ".co.uk", ".gov", ".io"]))):
            return self.predict_from_url(stripped, model_name=model_name, explain=explain)

        if not stripped:
            return {
                "label": "UNKNOWN",
                "label_code": -1,
                "confidence": 0.0,
                "confidence_band": "LOW",
                "confidence_source": "Input Empty",
                "probabilities": {"REAL": 0.5, "FAKE": 0.5},
                "explanation": "No text provided.",
                "preprocessed_text": "",
                "top_features": [],
                "sensationalism_score": 0.0,
                "sensational_keywords": [],
                "inspection": {},
                "explanation_features": None,
            }

        selected_model_name = model_name or self.current_model_name
        model = self.models.get(selected_model_name)
        if model is None:
            raise ValueError(f"Model '{selected_model_name}' not available.")

        # Step 1: Preprocessing
        preprocessed = preprocess_text(stripped)
        inspection = inspect_preprocessing_steps(stripped)

        if not preprocessed.strip():
            return {
                "label": "INSUFFICIENT TEXT",
                "label_code": -1,
                "confidence": 0.0,
                "confidence_band": "LOW",
                "confidence_source": "Text Insufficient",
                "probabilities": {"REAL": 0.5, "FAKE": 0.5},
                "explanation": "Text only contained stopwords or symbols.",
                "preprocessed_text": preprocessed,
                "top_features": [],
                "sensationalism_score": 0.0,
                "sensational_keywords": [],
                "inspection": inspection,
                "explanation_features": None,
            }

        # Step 2: TF-IDF Transformation
        tfidf_vec = self.vectorizer.transform([stripped])

        # Step 3: Model Inference & Probability Calculation
        pred_code = int(model.predict(tfidf_vec)[0])
        label = "FAKE" if pred_code == 1 else "REAL"

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(tfidf_vec)[0]
            real_prob = float(probs[0])
            fake_prob = float(probs[1])
            confidence = float(probs[pred_code])
        else:
            decision = float(model.decision_function(tfidf_vec)[0])
            fake_prob = float(1.0 / (1.0 + np.exp(-decision)))
            real_prob = 1.0 - fake_prob
            confidence = fake_prob if pred_code == 1 else real_prob

        # Step 4: Linguistic Sensationalism Analysis
        sens_score, sens_keywords = compute_sensationalism_score(stripped[:120], stripped)

        # Step 5: Extract Salient TF-IDF features
        top_features = get_top_features_for_text(self.vectorizer, stripped, top_n=8)

        # Step 6: Optional LIME Explainability
        explanation_features = None
        if explain:
            def predict_fn(texts: List[str]) -> np.ndarray:
                vecs = self.vectorizer.transform(texts)
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(vecs)
                else:
                    dec = model.decision_function(vecs)
                    fake_p = 1.0 / (1.0 + np.exp(-dec))
                    return np.column_stack([1.0 - fake_p, fake_p])

            explanation_features = self.explainer.explain_instance(
                text=stripped,
                predict_fn=predict_fn,
                num_features=8,
            )

        return {
            "label": label,
            "label_code": pred_code,
            "confidence": confidence,
            "confidence_band": self._get_confidence_band(confidence),
            "confidence_source": "NLP Machine Learning Model",
            "probabilities": {
                "REAL": real_prob,
                "FAKE": fake_prob,
            },
            "model_used": selected_model_name,
            "sensationalism_score": sens_score,
            "sensational_keywords": sens_keywords,
            "preprocessed_text": preprocessed,
            "top_features": top_features,
            "inspection": inspection,
            "explanation_features": explanation_features,
        }

    def predict_from_url(
        self,
        url: str,
        model_name: Optional[str] = None,
        explain: bool = False,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Fetch article content from live URL, evaluate domain credibility, and classify text."""
        # 1. Check in-memory LRU cache
        if use_cache:
            cached = GLOBAL_CACHE.get(url)
            if cached is not None:
                # If explain was specifically requested and not in cache, we can recompute or serve
                if not explain or cached.get("explanation_features"):
                    return cached

        scraped = extract_article_from_url(url)
        domain = scraped.get("domain", "")
        credibility = scraped.get("credibility", {})
        cred_status = credibility.get("status", "UNVERIFIED_DOMAIN")
        sens_score = scraped.get("sensationalism_score", 0.0)
        sens_keywords = scraped.get("sensational_keywords_found", [])
        title = scraped.get("title", "")
        body = scraped.get("content", "")

        # Fallback if scraping was completely blocked
        if not scraped.get("success"):
            if cred_status == "VERIFIED_OUTLET":
                final_label = "REAL"
                real_p, fake_p = 0.980, 0.020
                conf_band = "HIGH"
                conf_source = "Accredited Domain Whitelist"
            elif cred_status in ["KNOWN_HOAX", "KNOWN_SATIRE"]:
                final_label = "FAKE"
                real_p, fake_p = 0.050, 0.950
                conf_band = "HIGH"
                conf_source = "Disinformation Registry"
            else:
                final_label = "UNKNOWN"
                real_p, fake_p = 0.500, 0.500
                conf_band = "LOW"
                conf_source = "Unverified Domain / Scrape Blocked"

            res = {
                "success": False,
                "url": url,
                "domain": domain,
                "credibility": credibility,
                "label": final_label,
                "label_code": 0 if final_label == "REAL" else 1,
                "confidence": real_p if final_label == "REAL" else fake_p,
                "confidence_band": conf_band,
                "confidence_source": conf_source,
                "probabilities": {"REAL": real_p, "FAKE": fake_p},
                "article_title": title,
                "article_content_snippet": "",
                "word_count": 0,
                "sensationalism_score": 0.0,
                "sensational_keywords": [],
                "error_message": scraped.get("error_message"),
                "note": "Could not scrape full article body (website may require login or block crawlers).",
                "cached": False,
                "explanation_features": None,
            }
            if use_cache:
                GLOBAL_CACHE.set(url, res)
            return res

        # Run NLP prediction on extracted article text
        combined_text = f"{title}. {body}"
        nlp_result = self.predict(combined_text, model_name=model_name, explain=explain)

        # Multi-Tier Verification Logic
        if cred_status == "VERIFIED_OUTLET":
            # Accredited major publisher (BBC, Reuters, AP, The Guardian, The Hindu, etc.)
            if sens_score >= 0.70:
                final_label = "FAKE (High Sensationalism Alert)"
                pred_code = 1
                fake_prob = float(sens_score)
                real_prob = float(1.0 - fake_prob)
                confidence = fake_prob
                conf_band = self._get_confidence_band(confidence)
                conf_source = "Sensationalism Detector Override"
            else:
                final_label = "REAL"
                pred_code = 0
                real_prob = 0.985
                fake_prob = 0.015
                confidence = 0.985
                conf_band = "HIGH"
                conf_source = "Verified News Organization Editorial Standards"

        elif cred_status == "KNOWN_SATIRE":
            # Known parody/humor site (The Onion, Babylon Bee)
            final_label = "FAKE (Satire / Parody)"
            pred_code = 1
            fake_prob = 0.985
            real_prob = 0.015
            confidence = 0.985
            conf_band = "HIGH"
            conf_source = "Known Satirical / Parody Publication"

        elif cred_status == "KNOWN_HOAX":
            # Documented conspiracy/disinformation site
            final_label = "FAKE"
            pred_code = 1
            fake_prob = 0.990
            real_prob = 0.010
            confidence = 0.990
            conf_band = "HIGH"
            conf_source = "Documented Conspiracy / Fake News Platform"

        else:
            # Independent / Unverified Domain:
            # Driven by the high-accuracy NLP ML Model + Sensationalism scoring
            raw_fake = nlp_result["probabilities"]["FAKE"]
            raw_real = nlp_result["probabilities"]["REAL"]

            # Sensationalism amplifies fake probability
            combined_fake = np.clip(raw_fake * 0.7 + sens_score * 0.3, 0.05, 0.99)
            combined_real = 1.0 - combined_fake

            if combined_fake >= 0.50:
                final_label = "FAKE"
                pred_code = 1
                confidence = float(combined_fake)
            else:
                final_label = "REAL"
                pred_code = 0
                confidence = float(combined_real)

            real_prob = float(combined_real)
            fake_prob = float(combined_fake)
            conf_band = self._get_confidence_band(confidence)
            conf_source = "Hybrid NLP Model & Sensationalism Scoring"

        res = {
            "success": True,
            "url": url,
            "domain": domain,
            "credibility": credibility,
            "label": final_label,
            "label_code": pred_code,
            "confidence": confidence,
            "confidence_band": conf_band,
            "confidence_source": conf_source,
            "probabilities": {
                "REAL": real_prob,
                "FAKE": fake_prob,
            },
            "article_title": title,
            "article_content_snippet": body[:500] + ("..." if len(body) > 500 else ""),
            "word_count": scraped.get("word_count", 0),
            "sensationalism_score": sens_score,
            "sensational_keywords": sens_keywords,
            "model_used": nlp_result.get("model_used"),
            "top_features": nlp_result.get("top_features", []),
            "preprocessed_text": nlp_result.get("preprocessed_text", ""),
            "inspection": nlp_result.get("inspection", {}),
            "explanation_features": nlp_result.get("explanation_features"),
            "cached": False,
        }

        if use_cache:
            GLOBAL_CACHE.set(url, res)

        return res
