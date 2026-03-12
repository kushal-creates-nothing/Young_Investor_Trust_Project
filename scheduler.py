import logging
import os
import time
from datetime import datetime

import schedule

import config
from scraper.news_fetcher import NewsFetcher, _deduplicate
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
    logger.info("=== pipeline start %s ===", datetime.utcnow().isoformat())

    # 1. fetch
    news = NewsFetcher()
    rss = RSSFetcher()
    articles = _deduplicate(news.fetch_all() + rss.fetch_all_rss())
    logger.info("%d unique articles", len(articles))

    if not articles:
        logger.warning("nothing fetched — aborting")
        return {}

    # 2. sentiment
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_batch(articles)

    # 3. score
    scorer = MarketMoodScorer()
    mood = scorer.compute_mood_score(results)
    logger.info("mood: %s (%.3f)", mood.get("mood_label"), mood.get("overall_score", 0))

    # 4. store
    storage = DataStorage()
    url_map = storage.save_articles(articles)
    storage.save_sentiment_results(results, url_map)
    storage.save_mood_snapshot(mood)

    # 5. print summary
    reporter = ReportGenerator()
    print(reporter.generate_text_summary(mood, results))

    # 6. save reports
    os.makedirs("reports", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = f"reports/report_{ts}.json"
    with open(json_path, "w") as f:
        f.write(reporter.generate_json_report(mood, results))
    reporter.generate_html_snapshot(mood, results)

    logger.info("=== pipeline done ===")
    return mood


def main():
    interval = config.FETCH_INTERVAL_HOURS
    logger.info("scheduling every %d hour(s) — running now first...", interval)
    run_once()
    schedule.every(interval).hours.do(run_once)
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()

