import logging
from typing import List

import config
from sentiment.analyzer import SentimentResult

logger = logging.getLogger(__name__)

# compound score → mood label mapping, evaluated top-to-bottom
_THRESHOLDS = [
    (0.35, "Strongly Bullish"),
    (0.15, "Bullish"),
    (-0.15, "Neutral"),
    (-0.35, "Bearish"),
]


class MarketMoodScorer:

    def compute_mood_score(self, results: List[SentimentResult]) -> dict:
        total = len(results)
        if total == 0:
            return self._empty()

        sh_count = sum(1 for r in results if r.is_safe_haven_signal)
        ro_count = sum(1 for r in results if r.is_risk_on_signal)
        neg_count = sum(1 for r in results if r.vader_label == "negative")
        pos_count = sum(1 for r in results if r.vader_label == "positive")

        overall = sum(r.vader_compound for r in results) / total
        shpi = sh_count / total
        risk_on_idx = ro_count / total

        # fear = blend of negative articles and safe-haven mentions, capped at 1
        fear = min((neg_count + sh_count) / (total * 2), 1.0)

        mood_label = self._mood_label(overall, shpi)
        alert = self._tipping_point(shpi, overall)
        figures = self._notable_figures(results)

        return {
            "overall_score": round(overall, 4),
            "shpi": round(shpi, 4),
            "risk_on_index": round(risk_on_idx, 4),
            "fear_score": round(fear, 4),
            "mood_label": mood_label,
            "tipping_point_alert": alert,
            "article_count": total,
            "safe_haven_count": sh_count,
            "risk_on_count": ro_count,
            "negative_count": neg_count,
            "positive_count": pos_count,
            "notable_figures": figures,
        }

    def _mood_label(self, overall, shpi):
        if shpi >= config.SHPI_TIPPING_THRESHOLD and overall < config.SENTIMENT_TIPPING_THRESHOLD:
            return "Safe-Haven Flight"
        for threshold, label in _THRESHOLDS:
            if overall >= threshold:
                return label
        return "Strongly Bearish"

    def _tipping_point(self, shpi, overall):
        if shpi > config.SHPI_TIPPING_THRESHOLD and overall < config.SENTIMENT_TIPPING_THRESHOLD:
            return "⚠️ CAUTION: Market sentiment shifting toward safe havens"
        if shpi > config.SHPI_TIPPING_THRESHOLD * 0.75 and overall < 0.0:
            return "⚠️ TIPPING POINT APPROACHING — monitor closely"
        return ""

    def _notable_figures(self, results):
        data = {}
        for r in results:
            text = r.article.raw_text.lower()
            for fig in config.NOTABLE_FIGURES:
                if fig.lower() in text:
                    if fig not in data:
                        data[fig] = {"count": 0, "scores": []}
                    data[fig]["count"] += 1
                    data[fig]["scores"].append(r.vader_compound)

        out = []
        for fig, d in data.items():
            avg = sum(d["scores"]) / len(d["scores"])
            if avg >= 0.05:
                sent = "positive"
            elif avg <= -0.05:
                sent = "negative"
            else:
                sent = "neutral"
            out.append({
                "name": fig,
                "article_count": d["count"],
                "avg_sentiment": round(avg, 3),
                "sentiment_label": sent,
            })
        return out

    @staticmethod
    def _empty():
        return {
            "overall_score": 0.0, "shpi": 0.0, "risk_on_index": 0.0,
            "fear_score": 0.0, "mood_label": "Neutral", "tipping_point_alert": "",
            "article_count": 0, "safe_haven_count": 0, "risk_on_count": 0,
            "negative_count": 0, "positive_count": 0, "notable_figures": [],
        }
