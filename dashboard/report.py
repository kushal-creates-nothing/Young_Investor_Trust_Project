"""
dashboard/report.py — Report generator: JSON, terminal text, and HTML snapshots.

Designed for young investors: plain language, emojis, clear labels.
"""

import json
import logging
import os
from datetime import datetime
from typing import List

from jinja2 import Environment

from sentiment.analyzer import SentimentResult

logger = logging.getLogger(__name__)

REPORTS_DIR = "reports"

# ── Inline Jinja2 HTML template ───────────────────────────────────────────────
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Young Investor Market Mood Report</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }
    .card { background: white; border-radius: 12px; padding: 24px; max-width: 800px;
            margin: 0 auto; box-shadow: 0 2px 8px rgba(0,0,0,.1); }
    h1 { color: #1a1a2e; font-size: 1.6rem; }
    .metric { display: flex; justify-content: space-between; padding: 8px 0;
              border-bottom: 1px solid #eee; }
    .label { color: #555; }
    .value { font-weight: bold; }
    .alert { background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px;
             margin: 16px 0; border-radius: 4px; }
    .section-title { margin-top: 20px; font-size: 1.1rem; color: #333; }
    .headline-list { list-style: none; padding: 0; }
    .headline-list li { padding: 6px 0; border-bottom: 1px solid #f0f0f0; font-size: 0.9rem; }
    .neg { color: #dc3545; }
    .pos { color: #28a745; }
  </style>
</head>
<body>
<div class="card">
  <h1>🧠 Young Investor Market Mood Report</h1>
  <p>📅 {{ report_time }}</p>

  <div class="metric"><span class="label">Overall Sentiment</span>
    <span class="value">{{ mood_emoji }} {{ mood["mood_label"] }} ({{ mood["overall_score"] }})</span></div>
  <div class="metric"><span class="label">Safe-Haven Pressure Index</span>
    <span class="value">{{ shpi_emoji }} {{ mood["shpi"] }}</span></div>
  <div class="metric"><span class="label">Risk-On Activity</span>
    <span class="value">{{ risk_on_emoji }} {{ mood["risk_on_index"] }}</span></div>
  <div class="metric"><span class="label">Fear Score</span>
    <span class="value">{{ mood["fear_score"] }} / 1.00</span></div>
  <div class="metric"><span class="label">Articles Analysed</span>
    <span class="value">{{ mood["article_count"] }}</span></div>

  {% if mood["tipping_point_alert"] %}
  <div class="alert">{{ mood["tipping_point_alert"] }}</div>
  {% endif %}

  <p class="section-title">📰 Top Negative Headlines</p>
  <ul class="headline-list">
    {% for r in top_negative %}
    <li><span class="neg">({{ "%.2f"|format(r.vader_compound) }})</span> {{ r.article.title }}</li>
    {% endfor %}
  </ul>

  <p class="section-title">📰 Top Positive Headlines</p>
  <ul class="headline-list">
    {% for r in top_positive %}
    <li><span class="pos">(+{{ "%.2f"|format(r.vader_compound) }})</span> {{ r.article.title }}</li>
    {% endfor %}
  </ul>

  {% if mood["notable_figures"] %}
  <p class="section-title">👤 Notable Figures</p>
  <ul class="headline-list">
    {% for fig in mood["notable_figures"] %}
    <li><strong>{{ fig.name }}</strong> — {{ fig.sentiment_label }} context ({{ fig.article_count }} articles)</li>
    {% endfor %}
  </ul>
  {% endif %}
</div>
</body>
</html>"""


class ReportGenerator:
    """Generates JSON, text, and HTML reports from mood_dict + results."""

    def __init__(self, reports_dir: str = REPORTS_DIR):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    # ── JSON ───────────────────────────────────────────────────────────────────

    def generate_json_report(self, mood_dict: dict, results: List[SentimentResult]) -> str:
        """Return machine-readable JSON string."""
        top_neg = self._top_negative(results, 5)
        top_pos = self._top_positive(results, 5)

        report = {
            "report_time": datetime.utcnow().isoformat(),
            "mood": mood_dict,
            "top_negative_headlines": [
                {"title": r.article.title, "score": r.vader_compound, "source": r.article.source}
                for r in top_neg
            ],
            "top_positive_headlines": [
                {"title": r.article.title, "score": r.vader_compound, "source": r.article.source}
                for r in top_pos
            ],
            "safe_haven_articles": [
                r.article.title for r in results if r.is_safe_haven_signal
            ],
        }
        return json.dumps(report, indent=2, ensure_ascii=False)

    # ── Text ───────────────────────────────────────────────────────────────────

    def generate_text_summary(self, mood_dict: dict, results: List[SentimentResult]) -> str:
        """Return a human-readable terminal summary with emojis."""
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        mood_emoji = self._mood_emoji(mood_dict.get("overall_score", 0.0))
        shpi_emoji = self._shpi_emoji(mood_dict.get("shpi", 0.0))
        risk_on_emoji = self._risk_on_emoji(mood_dict.get("risk_on_index", 0.0))

        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║       🧠 YOUNG INVESTOR MARKET MOOD REPORT               ║",
            f"║       📅 {now:<47}║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Overall Sentiment:   {mood_emoji} {mood_dict.get('mood_label',''):<20} ({mood_dict.get('overall_score',0.0):+.2f})  ║",
            f"║  Safe-Haven Pressure: {shpi_emoji} {self._shpi_label(mood_dict.get('shpi',0.0)):<20} ({mood_dict.get('shpi',0.0):.2f})  ║",
            f"║  Risk-On Activity:    {risk_on_emoji} {self._risk_on_label(mood_dict.get('risk_on_index',0.0)):<20} ({mood_dict.get('risk_on_index',0.0):.2f})  ║",
            f"║  Fear Score:          {mood_dict.get('fear_score',0.0):.2f} / 1.00{' '*37}║",
            "╠══════════════════════════════════════════════════════════╣",
        ]

        alert = mood_dict.get("tipping_point_alert", "")
        if alert:
            lines.append(f"║  {alert:<55}║")
            lines.append("║  Market narrative is shifting toward gold & bonds       ║")

        lines.append("╚══════════════════════════════════════════════════════════╝")
        lines.append("")

        # Top negative
        top_neg = self._top_negative(results, 5)
        if top_neg:
            lines.append("📰 TOP NEGATIVE HEADLINES:")
            for i, r in enumerate(top_neg, 1):
                lines.append(f'  {i}. "{r.article.title[:70]}" ({r.vader_compound:.2f})')
            lines.append("")

        # Top positive
        top_pos = self._top_positive(results, 5)
        if top_pos:
            lines.append("📰 TOP POSITIVE HEADLINES:")
            for i, r in enumerate(top_pos, 1):
                lines.append(f'  {i}. "{r.article.title[:70]}" (+{r.vader_compound:.2f})')
            lines.append("")

        # Safe-haven count
        sh_count = mood_dict.get("safe_haven_count", 0)
        lines.append(f"🛡️  Safe-haven signal articles: {sh_count} / {mood_dict.get('article_count', 0)}")
        lines.append("")

        # Notable figures
        notable = mood_dict.get("notable_figures", [])
        if notable:
            lines.append("👤 NOTABLE FIGURES MENTIONED:")
            for fig in notable:
                lines.append(
                    f"  {fig['name']} — {fig['sentiment_label']} context ({fig['article_count']} articles)"
                )
            lines.append("")

        return "\n".join(lines)

    # ── HTML ───────────────────────────────────────────────────────────────────

    def generate_html_snapshot(self, mood_dict: dict, results: List[SentimentResult]) -> str:
        """Render HTML report, save to reports/ folder, and return the file path."""
        env = Environment(autoescape=True)
        template = env.from_string(_HTML_TEMPLATE)

        # Prepare notable figures as objects
        notable_raw = mood_dict.get("notable_figures", [])

        class _Figure:
            def __init__(self, d):
                self.name = d.get("name", "")
                self.sentiment_label = d.get("sentiment_label", "neutral")
                self.article_count = d.get("article_count", 0)

        notable_objs = [_Figure(f) for f in notable_raw]

        html = template.render(
            report_time=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            mood=mood_dict,
            mood_emoji=self._mood_emoji(mood_dict.get("overall_score", 0.0)),
            shpi_emoji=self._shpi_emoji(mood_dict.get("shpi", 0.0)),
            risk_on_emoji=self._risk_on_emoji(mood_dict.get("risk_on_index", 0.0)),
            top_negative=self._top_negative(results, 5),
            top_positive=self._top_positive(results, 5),
        )

        # Replace notable_figures placeholder (Jinja rendering handles it via mood dict)
        # Re-render with object list
        html = template.render(
            report_time=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            mood=mood_dict,
            mood_emoji=self._mood_emoji(mood_dict.get("overall_score", 0.0)),
            shpi_emoji=self._shpi_emoji(mood_dict.get("shpi", 0.0)),
            risk_on_emoji=self._risk_on_emoji(mood_dict.get("risk_on_index", 0.0)),
            top_negative=self._top_negative(results, 5),
            top_positive=self._top_positive(results, 5),
            notable_figures=notable_objs,
        )

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.reports_dir, f"report_{timestamp}.html")
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("HTML report saved: %s", filename)
        return filename

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _top_negative(results: List[SentimentResult], n: int) -> List[SentimentResult]:
        return sorted(results, key=lambda r: r.vader_compound)[:n]

    @staticmethod
    def _top_positive(results: List[SentimentResult], n: int) -> List[SentimentResult]:
        return sorted(results, key=lambda r: r.vader_compound, reverse=True)[:n]

    @staticmethod
    def _mood_emoji(score: float) -> str:
        if score >= 0.35:
            return "🟢"
        if score >= 0.05:
            return "🟡"
        if score >= -0.05:
            return "⚪"
        return "🔴"

    @staticmethod
    def _shpi_emoji(shpi: float) -> str:
        if shpi >= 0.4:
            return "🔴"
        if shpi >= 0.25:
            return "🟡"
        return "🟢"

    @staticmethod
    def _risk_on_emoji(roi: float) -> str:
        if roi >= 0.4:
            return "🟢"
        if roi >= 0.2:
            return "🟡"
        return "🔴"

    @staticmethod
    def _shpi_label(shpi: float) -> str:
        if shpi >= 0.4:
            return "High"
        if shpi >= 0.25:
            return "Moderate"
        return "Low"

    @staticmethod
    def _risk_on_label(roi: float) -> str:
        if roi >= 0.4:
            return "High"
        if roi >= 0.2:
            return "Moderate"
        return "Low"
