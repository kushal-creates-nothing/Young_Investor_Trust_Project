"""
scheduler.py — Main entry point for the Young Investor Trust Project.

Runs the full sentiment analysis pipeline every FETCH_INTERVAL_HOURS hours.
Pipeline steps per run:
  1. Fetch articles (NewsAPI + GNews + Finnhub + RSS fallback)
  2. Analyse sentiment on all articles
  3. Compute market mood score
  4. Save to SQLite database
  5. Print text summary to console
  6. Save JSON + HTML reports to reports/ folder
"""

import logging
import os
import time
from datetime import datetime

import schedule

import config
from scraper.news_fetcher import NewsFetcher
from scraper.rss_fetcher import RSSFetcher
from sentiment.analyzer import SentimentAnalyzer
from sentiment.scoring import MarketMoodScorer
from data.storage import DataStorage
from dashboard.report import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_once():
    """Execute the full pipeline once and return the mood_dict."""
    logger.info("=== Pipeline run started at %s ===", datetime.utcnow().isoformat())

    # 1. Fetch articles ─────────────────────────────────────────────────────────
    news_fetcher = NewsFetcher()
    rss_fetcher = RSSFetcher()

    api_articles = news_fetcher.fetch_all()
    rss_articles = rss_fetcher.fetch_all_rss()

    # Merge and deduplicate across sources
    from scraper.news_fetcher import _deduplicate
    all_articles = _deduplicate(api_articles + rss_articles)
    logger.info("Total unique articles fetched: %d", len(all_articles))

    if not all_articles:
        logger.warning("No articles fetched — pipeline aborted.")
        return {}

    # 2. Analyse sentiment ──────────────────────────────────────────────────────
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_batch(all_articles)
    logger.info("Sentiment analysis complete for %d articles.", len(results))

    # 3. Compute mood score ─────────────────────────────────────────────────────
    scorer = MarketMoodScorer()
    mood_dict = scorer.compute_mood_score(results)
    logger.info("Market mood: %s (score: %.3f)", mood_dict.get("mood_label"), mood_dict.get("overall_score", 0.0))

    # 4. Save to database ───────────────────────────────────────────────────────
    storage = DataStorage()
    url_to_id = storage.save_articles(all_articles)
    storage.save_sentiment_results(results, url_to_id)
    storage.save_mood_snapshot(mood_dict)

    # 5. Print text summary ─────────────────────────────────────────────────────
    reporter = ReportGenerator()
    text_summary = reporter.generate_text_summary(mood_dict, results)
    print(text_summary)

    # 6. Save JSON + HTML reports ───────────────────────────────────────────────
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    json_report = reporter.generate_json_report(mood_dict, results)
    json_path = os.path.join("reports", f"report_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(json_report)
    logger.info("JSON report saved: %s", json_path)

    html_path = reporter.generate_html_snapshot(mood_dict, results)
    logger.info("HTML report saved: %s", html_path)

    logger.info("=== Pipeline run complete ===")
    return mood_dict


def main():
    """Schedule and run the pipeline on the configured interval."""
    interval = config.FETCH_INTERVAL_HOURS
    logger.info("Scheduling pipeline every %d hour(s). Running immediately…", interval)

    # Run immediately on start
    run_once()

    # Then schedule every N hours
    schedule.every(interval).hours.do(run_once)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
