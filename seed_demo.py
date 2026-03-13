"""
Seed the database with realistic demo data so the dashboard works out of the box
without requiring live API keys. Run once: python seed_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import random

from scraper.news_fetcher import Article
from sentiment.analyzer import SentimentResult, SentimentAnalyzer
from sentiment.scoring import MarketMoodScorer
from data.storage import DataStorage
from dashboard.report import ReportGenerator

random.seed(42)

DEMO_ARTICLES = [
    # Bearish / safe-haven articles
    ("Fed signals further rate hikes amid persistent inflation", "The Federal Reserve indicated it may raise interest rates again as inflation remains stubbornly above targets.", "Reuters Business", -0.72, True, False),
    ("Global markets tumble on recession fears", "Stock markets around the world fell sharply as investors grow increasingly concerned about a global recession.", "BBC Business", -0.68, True, False),
    ("Gold hits 3-month high as investors seek safety", "Gold prices surged to a three-month high as uncertainty drives demand for safe-haven assets.", "MarketWatch", -0.22, True, False),
    ("Jerome Powell warns of prolonged economic uncertainty", "Federal Reserve chair Jerome Powell said the path back to 2% inflation would be bumpy.", "CNBC", -0.61, True, False),
    ("Bear market territory: S&P 500 drops 20% from peak", "The S&P 500 has officially entered bear market territory after a 20% decline from its January highs.", "Financial Times", -0.75, True, False),
    ("Recession fears mount as GDP contracts for second quarter", "Economists warn the economy may already be in recession after GDP shrank for a second consecutive quarter.", "The Guardian", -0.69, True, False),
    ("Sell-off accelerates as inflation data disappoints", "Wall Street suffered its worst day in months after the latest CPI reading came in hotter than expected.", "Bloomberg", -0.71, True, False),
    ("Warren Buffett increases cash reserves amid market uncertainty", "Berkshire Hathaway has been building its cash pile, signalling caution from the legendary investor.", "CNBC", -0.33, True, False),
    ("Central banks worldwide tighten policy as inflation spreads", "Multiple central banks raised rates this week as global inflation continues to prove resilient.", "Reuters", -0.55, True, False),
    ("IMF cuts global growth forecast citing multiple headwinds", "The IMF lowered its world growth outlook, citing trade tensions, inflation and rising debt levels.", "IMF", -0.60, True, False),
    # Bullish / risk-on articles
    ("Tech rally lifts markets as earnings beat expectations", "A strong earnings season in the technology sector pushed major indices higher on Thursday.", "CNBC", 0.65, False, True),
    ("GDP growth surprises to the upside in Q3", "The economy expanded at a faster-than-expected pace in the third quarter, easing recession fears.", "Reuters", 0.58, False, True),
    ("IPO market heats up with five major listings this week", "A flurry of high-profile IPOs signalled renewed investor optimism about equity markets.", "Bloomberg", 0.52, False, True),
    ("Earnings beat: Apple reports record quarterly revenue", "Apple posted its strongest ever quarter, with revenue and profit both exceeding analyst forecasts.", "MarketWatch", 0.71, False, True),
    ("Bull market confirmed as index closes 20% above trough", "Analysts declared a new bull market after the Nasdaq composite rose 20% from its October low.", "Financial Times", 0.74, False, True),
    ("Investment flows into equities reach six-month high", "Fund managers are rotating back into equities as confidence in the economic outlook improves.", "Reuters", 0.49, False, True),
    ("Elon Musk announces new AI venture, stocks rally", "Technology stocks surged after Elon Musk unveiled plans for a new artificial intelligence company.", "TechCrunch", 0.61, False, True),
    # Neutral / mixed
    ("Janet Yellen meets G7 finance ministers on debt ceiling", "Treasury Secretary Janet Yellen held talks with G7 counterparts amid ongoing US debt ceiling negotiations.", "AP", 0.03, False, False),
    ("World Bank releases annual development report", "The World Bank published its annual outlook, projecting moderate growth for developing economies.", "World Bank", 0.08, False, False),
    ("Federal Reserve meeting minutes released", "Minutes from the last FOMC meeting showed policymakers divided on the pace of future rate decisions.", "Reuters", -0.04, False, False),
]


def build_articles_and_results():
    analyzer = SentimentAnalyzer()
    articles = []
    results = []

    for i, (title, desc, source, score, is_safe, is_risk) in enumerate(DEMO_ARTICLES):
        art = Article(
            title=title,
            description=desc,
            url=f"https://demo.example.com/article/{i+1}",
            source=source,
            published_at=(datetime.utcnow() - timedelta(hours=random.randint(0, 4))).isoformat(),
            raw_text=f"{title} {desc}",
        )
        articles.append(art)

        label = "positive" if score >= 0.05 else ("negative" if score <= -0.05 else "neutral")
        kw = []
        if is_safe:
            import config
            kw = [k for k in config.RISK_OFF_KEYWORDS if k.lower() in art.raw_text.lower()]
        if is_risk:
            import config
            kw += [k for k in config.RISK_ON_KEYWORDS if k.lower() in art.raw_text.lower()]

        results.append(SentimentResult(
            article=art,
            vader_compound=score,
            vader_label=label,
            finbert_label=label,
            finbert_score=abs(score),
            keywords_matched=kw,
            is_safe_haven_signal=is_safe,
            is_risk_on_signal=is_risk,
        ))

    return articles, results


def seed():
    print("Seeding demo data...")
    articles, results = build_articles_and_results()

    storage = DataStorage()
    url_map = storage.save_articles(articles)
    storage.save_sentiment_results(results, url_map)

    # Create several historical snapshots to populate the trend chart
    scorer = MarketMoodScorer()

    # simulate 12 past snapshots with some variation
    from sqlalchemy.orm import Session
    from data.storage import MoodSnapshotModel
    import json

    shifts = [
        (0.15, 0.22, 0.31), (-0.05, 0.28, 0.38), (-0.12, 0.31, 0.44),
        (-0.19, 0.35, 0.46), (-0.24, 0.41, 0.42), (-0.31, 0.44, 0.39),
        (-0.28, 0.42, 0.41), (-0.22, 0.38, 0.35), (-0.17, 0.33, 0.32),
        (-0.09, 0.29, 0.28), (-0.04, 0.25, 0.24), (-0.02, 0.22, 0.21),
    ]

    from data.storage import Base
    from sqlalchemy import create_engine
    import config

    engine = create_engine(f"sqlite:///{config.DB_PATH}", echo=False)
    with Session(engine) as session:
        for idx, (overall, shpi, fear) in enumerate(shifts):
            snap_time = datetime.utcnow() - timedelta(hours=(len(shifts) - idx) * 2)
            if overall >= 0.35:
                label = "Strongly Bullish"
            elif overall >= 0.15:
                label = "Bullish"
            elif overall >= -0.15:
                label = "Neutral"
            elif overall >= -0.35:
                label = "Bearish"
            elif shpi >= 0.4 and overall < -0.2:
                label = "Safe-Haven Flight"
            else:
                label = "Strongly Bearish"

            alert = ""
            if shpi > 0.4 and overall < -0.2:
                alert = "⚠️ CAUTION: Market sentiment shifting toward safe havens"

            session.add(MoodSnapshotModel(
                snapshot_time=snap_time,
                overall_score=overall,
                shpi=shpi,
                risk_on_index=round(random.uniform(0.10, 0.25), 3),
                fear_score=fear,
                mood_label=label,
                tipping_point_alert=alert,
                article_count=len(articles),
            ))
        session.commit()

    # save the current snapshot
    mood = scorer.compute_mood_score(results)
    storage.save_mood_snapshot(mood)

    # generate a sample HTML report
    reporter = ReportGenerator()
    reporter.generate_html_snapshot(mood, results)

    print(f"\nSeeded {len(articles)} demo articles + 13 mood snapshots.")
    print("Mood label:", mood["mood_label"])
    print("Overall score:", mood["overall_score"])
    print("\nStart the web server with:  python app.py")
    print("Then open:  http://localhost:5000\n")


if __name__ == "__main__":
    seed()
