"""
sentiment/scoring.py — Aggregate scoring and Safe-Haven Pressure Index.

Converts a list of SentimentResult objects into a market mood snapshot dict.
"""

import logging
from typing import List

import config
from sentiment.analyzer import SentimentResult

logger = logging.getLogger(__name__)

# Mood label thresholds (based on overall VADER compound)
_MOOD_THRESHOLDS = [
    (0.35, "Strongly Bullish"),
    (0.15, "Bullish"),
    (-0.15, "Neutral"),
    (-0.35, "Bearish"),
]

NOTABLE_FIGURES_LOWER = [f.lower() for f in config.NOTABLE_FIGURES]


class MarketMoodScorer:
    """Computes market mood metrics from a list of SentimentResult objects."""

    def compute_mood_score(self, results: List[SentimentResult]) -> dict:
        """
        Compute aggregate mood metrics.

        Returns a dict with keys:
            overall_score, shpi, risk_on_index, fear_score,
            mood_label, tipping_point_alert, article_count,
            safe_haven_count, risk_on_count, negative_count, positive_count,
            notable_figures
        """
        total = len(results)
        if total == 0:
            return self._empty_snapshot()

        safe_haven_count = sum(1 for r in results if r.is_safe_haven_signal)
        risk_on_count = sum(1 for r in results if r.is_risk_on_signal)
        negative_count = sum(1 for r in results if r.vader_label == "negative")
        positive_count = sum(1 for r in results if r.vader_label == "positive")

        # Overall sentiment: weighted average of VADER compound scores
        overall_score = sum(r.vader_compound for r in results) / total

        # Safe-Haven Pressure Index: ratio of safe-haven articles
        shpi = safe_haven_count / total

        # Risk-On Index: ratio of risk-on articles
        risk_on_index = risk_on_count / total

        # Fear vs Greed score: higher = more fear
        fear_score = (negative_count + safe_haven_count) / (total * 2) if total > 0 else 0.0
        fear_score = min(fear_score, 1.0)

        # Mood label
        mood_label = self._compute_mood_label(overall_score, shpi)

        # Tipping point alert
        tipping_point_alert = self._check_tipping_point(shpi, overall_score)

        # Notable figures analysis
        notable_figures = self._analyse_notable_figures(results)

        return {
            "overall_score": round(overall_score, 4),
            "shpi": round(shpi, 4),
            "risk_on_index": round(risk_on_index, 4),
            "fear_score": round(fear_score, 4),
            "mood_label": mood_label,
            "tipping_point_alert": tipping_point_alert,
            "article_count": total,
            "safe_haven_count": safe_haven_count,
            "risk_on_count": risk_on_count,
            "negative_count": negative_count,
            "positive_count": positive_count,
            "notable_figures": notable_figures,
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _compute_mood_label(self, overall_score: float, shpi: float) -> str:
        """Map overall sentiment + SHPI to a mood label."""
        # Special case: even if overall sentiment is slightly positive, a very
        # high SHPI indicates flight-to-safety behaviour
        if shpi >= config.SHPI_TIPPING_THRESHOLD and overall_score < config.SENTIMENT_TIPPING_THRESHOLD:
            return "Safe-Haven Flight"

        for threshold, label in _MOOD_THRESHOLDS:
            if overall_score >= threshold:
                return label

        return "Strongly Bearish"

    def _check_tipping_point(self, shpi: float, overall_score: float) -> str:
        """Return tipping-point alert string if threshold exceeded."""
        if shpi > config.SHPI_TIPPING_THRESHOLD and overall_score < config.SENTIMENT_TIPPING_THRESHOLD:
            return "⚠️ CAUTION: Market sentiment shifting toward safe havens"
        if shpi > config.SHPI_TIPPING_THRESHOLD * 0.75 and overall_score < 0.0:
            return "⚠️ TIPPING POINT APPROACHING — monitor closely"
        return ""

    def _analyse_notable_figures(self, results: List[SentimentResult]) -> List[dict]:
        """Detect notable figures in articles and summarise their sentiment context."""
        figure_data: dict = {}
        for result in results:
            text_lower = result.article.raw_text.lower()
            for figure in config.NOTABLE_FIGURES:
                if figure.lower() in text_lower:
                    if figure not in figure_data:
                        figure_data[figure] = {"count": 0, "compounds": []}
                    figure_data[figure]["count"] += 1
                    figure_data[figure]["compounds"].append(result.vader_compound)

        output = []
        for figure, data in figure_data.items():
            compounds = data["compounds"]
            avg = sum(compounds) / len(compounds)
            if avg >= 0.05:
                sentiment = "positive"
            elif avg <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            output.append(
                {
                    "name": figure,
                    "article_count": data["count"],
                    "avg_sentiment": round(avg, 3),
                    "sentiment_label": sentiment,
                }
            )
        return output

    @staticmethod
    def _empty_snapshot() -> dict:
        return {
            "overall_score": 0.0,
            "shpi": 0.0,
            "risk_on_index": 0.0,
            "fear_score": 0.0,
            "mood_label": "Neutral",
            "tipping_point_alert": "",
            "article_count": 0,
            "safe_haven_count": 0,
            "risk_on_count": 0,
            "negative_count": 0,
            "positive_count": 0,
            "notable_figures": [],
        }
