import json
import logging
import os
import threading
from datetime import datetime

from flask import Flask, jsonify, render_template, request

import config
from data.storage import DataStorage
from scraper.news_fetcher import NewsFetcher, _deduplicate
from scraper.rss_fetcher import RSSFetcher
from sentiment.analyzer import SentimentAnalyzer
from sentiment.scoring import MarketMoodScorer
from dashboard.report import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# track whether a run is in progress so we don't double-trigger
_run_lock = threading.Lock()
_pipeline_running = False
_last_run_time = None
_cached_results = []  # keep latest results in memory for quick API responses


def _run_pipeline():
    global _pipeline_running, _last_run_time, _cached_results

    logger.info("Pipeline starting...")
    news = NewsFetcher()
    rss = RSSFetcher()

    api_arts = news.fetch_all()
    rss_arts = rss.fetch_all_rss()
    articles = _deduplicate(api_arts + rss_arts)

    if not articles:
        logger.warning("No articles fetched — nothing to analyse")
        _pipeline_running = False
        return {}

    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_batch(articles)

    scorer = MarketMoodScorer()
    mood = scorer.compute_mood_score(results)

    storage = DataStorage()
    url_map = storage.save_articles(articles)
    storage.save_sentiment_results(results, url_map)
    storage.save_mood_snapshot(mood)

    reporter = ReportGenerator()
    os.makedirs("reports", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = f"reports/report_{ts}.json"
    with open(json_path, "w") as f:
        f.write(reporter.generate_json_report(mood, results))
    reporter.generate_html_snapshot(mood, results)

    _cached_results = results
    _last_run_time = datetime.utcnow().isoformat()
    _pipeline_running = False
    logger.info("Pipeline complete — mood: %s", mood.get("mood_label"))
    return mood


# ── routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/latest")
def api_latest():
    try:
        storage = DataStorage()
        snaps = storage.get_last_n_snapshots(1)
        if not snaps:
            return jsonify({"error": "No data yet — run the pipeline first"}), 404
        snap = snaps[0]
        # attach top headlines from memory cache if available
        headlines = _build_headlines(_cached_results)
        snap["headlines"] = headlines
        snap["last_run_time"] = _last_run_time
        return jsonify(snap)
    except Exception as e:
        logger.exception("Error in /api/latest")
        return jsonify({"error": str(e)}), 500


@app.route("/api/history")
def api_history():
    n = request.args.get("n", 24, type=int)
    try:
        storage = DataStorage()
        snaps = storage.get_last_n_snapshots(n)
        # reverse so oldest first for charting
        snaps.reverse()
        return jsonify(snaps)
    except Exception as e:
        logger.exception("Error in /api/history")
        return jsonify({"error": str(e)}), 500


@app.route("/api/headlines")
def api_headlines():
    if not _cached_results:
        return jsonify({"negative": [], "positive": [], "safe_haven": []})
    return jsonify(_build_headlines(_cached_results))


@app.route("/api/run", methods=["POST"])
def api_run():
    global _pipeline_running
    with _run_lock:
        if _pipeline_running:
            return jsonify({"status": "already_running", "message": "Analysis already in progress"}), 409
        _pipeline_running = True

    # run in background thread so the response returns immediately
    t = threading.Thread(target=_run_pipeline, daemon=True)
    t.start()
    return jsonify({"status": "started", "message": "Analysis started — check back in ~30 seconds"})


@app.route("/api/status")
def api_status():
    return jsonify({
        "pipeline_running": _pipeline_running,
        "last_run_time": _last_run_time,
        "articles_in_memory": len(_cached_results),
    })


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_headlines(results, n=7):
    if not results:
        return {"negative": [], "positive": [], "safe_haven": []}
    sorted_neg = sorted(results, key=lambda r: r.vader_compound)[:n]
    sorted_pos = sorted(results, key=lambda r: r.vader_compound, reverse=True)[:n]
    safe = [r for r in results if r.is_safe_haven_signal][:n]

    def fmt(r):
        return {
            "title": r.article.title,
            "score": round(r.vader_compound, 3),
            "source": r.article.source,
            "url": r.article.url,
        }

    return {
        "negative": [fmt(r) for r in sorted_neg],
        "positive": [fmt(r) for r in sorted_pos],
        "safe_haven": [fmt(r) for r in safe],
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
