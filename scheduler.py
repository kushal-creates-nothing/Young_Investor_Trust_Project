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
from mlops.wandb_tracker import SentimentRunTracker
from mlops.drift_monitor import DriftMonitor
from rag.vector_store import ArticleVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# module-level singletons so we don't reload models on every run
_vector_store = None
_drift_monitor = DriftMonitor()


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        try:
            _vector_store = ArticleVectorStore(persist_dir=config.CHROMA_PERSIST_DIR)
        except Exception as e:
            logger.warning("Vector store unavailable: %s", e)
    return _vector_store


def run_once():
    logger.info("=== pipeline start %s ===", datetime.utcnow().isoformat())
    tracker = SentimentRunTracker(project=config.WANDB_PROJECT)
    tracker.start_run(config={
        "model": "VADER+FinBERT" if config.USE_FINBERT else "VADER",
        "interval_hours": config.FETCH_INTERVAL_HOURS,
        "finbert_backend": "triton" if config.USE_TRITON else "local",
        "embedding_model": config.EMBEDDING_MODEL,
    })

    # 1. fetch
    news = NewsFetcher()
    rss = RSSFetcher()
    articles = _deduplicate(news.fetch_all() + rss.fetch_all_rss())
    logger.info("%d unique articles", len(articles))

    if not articles:
        logger.warning("nothing fetched — aborting")
        tracker.finish()
        return {}

    tracker.log_dataset_stats(articles)

    # 2. sentiment
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_batch(articles)

    # 3. score
    scorer = MarketMoodScorer()
    mood = scorer.compute_mood_score(results)
    logger.info("mood: %s (%.3f)", mood.get("mood_label"), mood.get("overall_score", 0))

    tracker.log_metrics(mood)

    # 4. store in SQLite
    storage = DataStorage()
    url_map = storage.save_articles(articles)
    storage.save_sentiment_results(results, url_map)
    storage.save_mood_snapshot(mood)

    # 5. drift detection — compare last 24 snapshots
    snapshots = storage.get_last_n_snapshots(24)
    if len(snapshots) >= 4:
        drift_report = _drift_monitor.check_from_snapshots(snapshots)
        mood["drift_report"] = drift_report
        if drift_report.get("overall_status") == "warning":
            logger.warning("Drift detected: %s", drift_report)
        if tracker._active:
            drift_metrics = drift_report.get("sentiment_drift", {})
            if drift_metrics.get("psi") is not None:
                tracker.log_metrics({"drift_psi": drift_metrics["psi"],
                                     "drift_ks_stat": drift_metrics.get("ks_statistic", 0)})

    # 6. embed articles into vector store for RAG
    vs = _get_vector_store()
    if vs:
        added = vs.add_articles(articles, results)
        logger.info("Vector store: %d articles indexed (total: %d)", added, vs.count())

    # 7. print summary
    reporter = ReportGenerator()
    print(reporter.generate_text_summary(mood, results))

    # 8. save reports
    os.makedirs("reports", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = f"reports/report_{ts}.json"
    with open(json_path, "w") as f:
        f.write(reporter.generate_json_report(mood, results))
    reporter.generate_html_snapshot(mood, results)

    tracker.finish()
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

