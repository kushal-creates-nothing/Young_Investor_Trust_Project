import json
import logging
import os
from datetime import datetime
from typing import List

from jinja2 import Environment

from sentiment.analyzer import SentimentResult

logger = logging.getLogger(__name__)

REPORTS_DIR = "reports"

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Market Mood — {{ report_time }}</title>
  <style>
    body { font-family: Arial, sans-serif; background:#0f172a; color:#e2e8f0; margin:0; padding:20px; }
    .card { background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1);
            border-radius:12px; padding:24px; max-width:860px; margin:0 auto 20px; }
    h1 { margin:0 0 4px; font-size:1.4rem; }
    .sub { color:#94a3b8; font-size:.85rem; margin-bottom:20px; }
    .row { display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid rgba(255,255,255,.07); }
    .lbl { color:#94a3b8; }
    .val { font-weight:600; }
    .alert { background:rgba(251,146,60,.12); border-left:3px solid #f97316;
             padding:12px 16px; border-radius:6px; margin:16px 0; color:#fdba74; }
    h3 { margin:20px 0 8px; color:#c7d2fe; font-size:.9rem; text-transform:uppercase; letter-spacing:.05em; }
    ul { list-style:none; padding:0; margin:0; }
    li { padding:6px 0; border-bottom:1px solid rgba(255,255,255,.05); font-size:.88rem; }
    .neg { color:#f87171; }
    .pos { color:#4ade80; }
    .src { color:#64748b; font-size:.78rem; }
  </style>
</head>
<body>
<div class="card">
  <h1>🧠 Young Investor Market Mood Report</h1>
  <p class="sub">📅 {{ report_time }}</p>

  <div class="row"><span class="lbl">Overall Sentiment</span>
    <span class="val">{{ mood_emoji }} {{ mood["mood_label"] }} ({{ mood["overall_score"] }})</span></div>
  <div class="row"><span class="lbl">Safe-Haven Pressure Index</span>
    <span class="val">{{ shpi_emoji }} {{ "%.0f"|format(mood["shpi"]*100) }}%</span></div>
  <div class="row"><span class="lbl">Risk-On Activity</span>
    <span class="val">{{ risk_on_emoji }} {{ "%.0f"|format(mood["risk_on_index"]*100) }}%</span></div>
  <div class="row"><span class="lbl">Fear Score</span>
    <span class="val">{{ "%.0f"|format(mood["fear_score"]*100) }}%</span></div>
  <div class="row"><span class="lbl">Articles Analysed</span>
    <span class="val">{{ mood["article_count"] }}</span></div>

  {% if mood["tipping_point_alert"] %}
  <div class="alert">{{ mood["tipping_point_alert"] }}</div>
  {% endif %}

  <h3>📰 Most Bearish Headlines</h3>
  <ul>
    {% for r in top_negative %}
    <li>
      <span class="neg">({{ "%.2f"|format(r.vader_compound) }})</span>
      {{ r.article.title }}
      <span class="src">— {{ r.article.source }}</span>
    </li>
    {% endfor %}
  </ul>

  <h3>📰 Most Bullish Headlines</h3>
  <ul>
    {% for r in top_positive %}
    <li>
      <span class="pos">(+{{ "%.2f"|format(r.vader_compound) }})</span>
      {{ r.article.title }}
      <span class="src">— {{ r.article.source }}</span>
    </li>
    {% endfor %}
  </ul>

  {% if notable_figures %}
  <h3>👤 Notable Figures</h3>
  <ul>
    {% for fig in notable_figures %}
    <li><strong>{{ fig.name }}</strong> — {{ fig.sentiment_label }} ({{ fig.article_count }} articles)</li>
    {% endfor %}
  </ul>
  {% endif %}
</div>
</body>
</html>"""


class ReportGenerator:

    def __init__(self, reports_dir=REPORTS_DIR):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    def generate_json_report(self, mood, results: List[SentimentResult]) -> str:
        return json.dumps({
            "report_time": datetime.utcnow().isoformat(),
            "mood": mood,
            "top_negative": [
                {"title": r.article.title, "score": r.vader_compound, "source": r.article.source}
                for r in self._top_neg(results, 5)
            ],
            "top_positive": [
                {"title": r.article.title, "score": r.vader_compound, "source": r.article.source}
                for r in self._top_pos(results, 5)
            ],
            "safe_haven_articles": [r.article.title for r in results if r.is_safe_haven_signal],
        }, indent=2, ensure_ascii=False)

    def generate_text_summary(self, mood, results: List[SentimentResult]) -> str:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        score = mood.get("overall_score", 0.0)
        shpi = mood.get("shpi", 0.0)
        roi = mood.get("risk_on_index", 0.0)

        me = "🟢" if score >= 0.15 else ("🔴" if score <= -0.15 else "🟡")
        se = "🔴" if shpi >= 0.4 else ("🟡" if shpi >= 0.25 else "🟢")
        re = "🟢" if roi >= 0.4 else ("🟡" if roi >= 0.2 else "🔴")

        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║       🧠 YOUNG INVESTOR MARKET MOOD REPORT               ║",
            f"║       📅 {now:<47}║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Overall Sentiment:   {me} {mood.get('mood_label',''):<22} ({score:+.2f}) ║",
            f"║  Safe-Haven Pressure: {se} {('High' if shpi>=0.4 else 'Moderate' if shpi>=0.25 else 'Low'):<22} ({shpi:.2f}) ║",
            f"║  Risk-On Activity:    {re} {('High' if roi>=0.4 else 'Moderate' if roi>=0.2 else 'Low'):<22} ({roi:.2f}) ║",
            f"║  Fear Score:          {mood.get('fear_score',0.0):.2f} / 1.00                                   ║",
            "╠══════════════════════════════════════════════════════════╣",
        ]
        alert = mood.get("tipping_point_alert", "")
        if alert:
            lines.append(f"║  {alert:<55}║")
            lines.append("║  Market narrative is shifting toward gold & bonds       ║")
        lines.append("╚══════════════════════════════════════════════════════════╝\n")

        neg5 = self._top_neg(results, 5)
        if neg5:
            lines.append("📰 TOP NEGATIVE HEADLINES:")
            for i, r in enumerate(neg5, 1):
                lines.append(f'  {i}. "{r.article.title[:72]}" ({r.vader_compound:.2f})')
            lines.append("")

        pos5 = self._top_pos(results, 5)
        if pos5:
            lines.append("📰 TOP POSITIVE HEADLINES:")
            for i, r in enumerate(pos5, 1):
                lines.append(f'  {i}. "{r.article.title[:72]}" (+{r.vader_compound:.2f})')
            lines.append("")

        lines.append(f"🛡️  Safe-haven signals: {mood.get('safe_haven_count',0)} / {mood.get('article_count',0)} articles\n")

        figures = mood.get("notable_figures", [])
        if figures:
            lines.append("👤 NOTABLE FIGURES MENTIONED:")
            for fig in figures:
                lines.append(f"  {fig['name']} — {fig['sentiment_label']} context ({fig['article_count']} articles)")
            lines.append("")

        return "\n".join(lines)

    def generate_html_snapshot(self, mood, results: List[SentimentResult]) -> str:
        env = Environment(autoescape=True)
        tmpl = env.from_string(_HTML_TEMPLATE)

        class _Fig:
            def __init__(self, d):
                self.name = d.get("name", "")
                self.sentiment_label = d.get("sentiment_label", "neutral")
                self.article_count = d.get("article_count", 0)

        score = mood.get("overall_score", 0.0)
        shpi = mood.get("shpi", 0.0)
        roi = mood.get("risk_on_index", 0.0)

        html = tmpl.render(
            report_time=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            mood=mood,
            mood_emoji="🟢" if score >= 0.15 else ("🔴" if score <= -0.15 else "🟡"),
            shpi_emoji="🔴" if shpi >= 0.4 else ("🟡" if shpi >= 0.25 else "🟢"),
            risk_on_emoji="🟢" if roi >= 0.4 else ("🟡" if roi >= 0.2 else "🔴"),
            top_negative=self._top_neg(results, 5),
            top_positive=self._top_pos(results, 5),
            notable_figures=[_Fig(f) for f in mood.get("notable_figures", [])],
        )

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.reports_dir, f"report_{ts}.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("HTML report: %s", path)
        return path

    @staticmethod
    def _top_neg(results, n):
        return sorted(results, key=lambda r: r.vader_compound)[:n]

    @staticmethod
    def _top_pos(results, n):
        return sorted(results, key=lambda r: r.vader_compound, reverse=True)[:n]

