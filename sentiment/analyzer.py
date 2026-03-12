"""
sentiment/analyzer.py — Sentiment analysis using VADER + optional FinBERT.

Primary: vaderSentiment (fast, good for short news headlines).
Secondary: HuggingFace ProsusAI/finbert (financial domain NLP, opt-in via USE_FINBERT=true).
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config
from scraper.news_fetcher import Article

logger = logging.getLogger(__name__)

# Lazy-load FinBERT to avoid import overhead when not needed
_finbert_pipeline = None


def _get_finbert():
    """Load the FinBERT pipeline on first use."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            _finbert_pipeline = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=1,
            )
            logger.info("FinBERT pipeline loaded.")
        except Exception as exc:
            logger.error("Failed to load FinBERT: %s", exc)
            _finbert_pipeline = None
    return _finbert_pipeline


@dataclass
class SentimentResult:
    article: Article
    vader_compound: float       # -1.0 to 1.0
    vader_label: str            # "positive", "negative", "neutral"
    finbert_label: str          # "positive", "negative", "neutral" (if enabled)
    finbert_score: float        # confidence score (0.0 if disabled)
    keywords_matched: List[str]
    is_safe_haven_signal: bool
    is_risk_on_signal: bool


class SentimentAnalyzer:
    """Performs sentiment analysis on Article objects."""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.use_finbert = config.USE_FINBERT
        # Normalise keyword lists to lower-case for matching
        self.risk_off_kw = [k.lower() for k in config.RISK_OFF_KEYWORDS]
        self.risk_on_kw = [k.lower() for k in config.RISK_ON_KEYWORDS]

    # ── VADER ──────────────────────────────────────────────────────────────────

    def _vader_score(self, text: str) -> tuple[float, str]:
        """Return (compound, label) from VADER analysis."""
        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return compound, label

    # ── FinBERT ────────────────────────────────────────────────────────────────

    def _finbert_score(self, text: str) -> tuple[str, float]:
        """Return (label, confidence) from FinBERT. Falls back to ('neutral', 0.0)."""
        if not self.use_finbert:
            return "neutral", 0.0
        pipe = _get_finbert()
        if pipe is None:
            return "neutral", 0.0
        try:
            # Truncate to 512 chars to avoid tokeniser limit issues
            result = pipe(text[:512])
            # result is list[list[dict]] when top_k=1
            if result and result[0]:
                top = result[0][0]
                return top.get("label", "neutral").lower(), float(top.get("score", 0.0))
        except Exception as exc:
            logger.warning("FinBERT inference failed: %s", exc)
        return "neutral", 0.0

    # ── Keyword matching ───────────────────────────────────────────────────────

    def _match_keywords(self, text: str) -> tuple[List[str], bool, bool]:
        """Return (matched_keywords, is_safe_haven, is_risk_on)."""
        text_lower = text.lower()
        matched = []
        is_safe_haven = False
        is_risk_on = False

        for kw in self.risk_off_kw:
            if kw in text_lower:
                matched.append(kw)
                is_safe_haven = True

        for kw in self.risk_on_kw:
            if kw in text_lower:
                matched.append(kw)
                is_risk_on = True

        return matched, is_safe_haven, is_risk_on

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, article: Article) -> SentimentResult:
        """Analyze a single Article and return a SentimentResult."""
        text = article.raw_text or f"{article.title} {article.description}"
        vader_compound, vader_label = self._vader_score(text)
        finbert_label, finbert_score = self._finbert_score(text)
        keywords_matched, is_safe_haven, is_risk_on = self._match_keywords(text)

        return SentimentResult(
            article=article,
            vader_compound=vader_compound,
            vader_label=vader_label,
            finbert_label=finbert_label,
            finbert_score=finbert_score,
            keywords_matched=keywords_matched,
            is_safe_haven_signal=is_safe_haven,
            is_risk_on_signal=is_risk_on,
        )

    def analyze_batch(self, articles: List[Article]) -> List[SentimentResult]:
        """Analyze a list of Articles and return a list of SentimentResults."""
        return [self.analyze(a) for a in articles]
