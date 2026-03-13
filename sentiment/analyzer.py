import logging
from dataclasses import dataclass, field
from typing import List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config
from scraper.news_fetcher import Article

logger = logging.getLogger(__name__)

_finbert_pipeline = None


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            _finbert_pipeline = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=1,
            )
            logger.info("FinBERT loaded")
        except Exception as e:
            logger.error("Couldn't load FinBERT: %s", e)
    return _finbert_pipeline


@dataclass
class SentimentResult:
    article: Article
    vader_compound: float
    vader_label: str
    finbert_label: str
    finbert_score: float
    keywords_matched: List[str]
    is_safe_haven_signal: bool
    is_risk_on_signal: bool


class SentimentAnalyzer:

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.use_finbert = config.USE_FINBERT
        self.risk_off_kw = [k.lower() for k in config.RISK_OFF_KEYWORDS]
        self.risk_on_kw = [k.lower() for k in config.RISK_ON_KEYWORDS]

    def _vader_score(self, text):
        scores = self.vader.polarity_scores(text)
        c = scores["compound"]
        if c >= 0.05:
            label = "positive"
        elif c <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return c, label

    def _finbert_score(self, text):
        if not self.use_finbert:
            return "neutral", 0.0
        pipe = _get_finbert()
        if pipe is None:
            return "neutral", 0.0
        try:
            result = pipe(text[:512])
            if result and result[0]:
                top = result[0][0]
                return top.get("label", "neutral").lower(), float(top.get("score", 0.0))
        except Exception as e:
            logger.warning("FinBERT inference error: %s", e)
        return "neutral", 0.0

    def _match_keywords(self, text):
        tl = text.lower()
        matched = []
        is_safe = False
        is_risk = False
        for kw in self.risk_off_kw:
            if kw in tl:
                matched.append(kw)
                is_safe = True
        for kw in self.risk_on_kw:
            if kw in tl:
                matched.append(kw)
                is_risk = True
        return matched, is_safe, is_risk

    def analyze(self, article: Article) -> SentimentResult:
        text = article.raw_text or f"{article.title} {article.description}"
        vc, vl = self._vader_score(text)
        fl, fs = self._finbert_score(text)
        kw, is_safe, is_risk = self._match_keywords(text)
        return SentimentResult(
            article=article,
            vader_compound=vc,
            vader_label=vl,
            finbert_label=fl,
            finbert_score=fs,
            keywords_matched=kw,
            is_safe_haven_signal=is_safe,
            is_risk_on_signal=is_risk,
        )

    def analyze_batch(self, articles):
        return [self.analyze(a) for a in articles]
