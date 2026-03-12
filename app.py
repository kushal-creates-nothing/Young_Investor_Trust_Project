import json
import logging
import os
import threading
from datetime import datetime

from flask import Flask, jsonify, render_template, request

import config
from data.storage import DataStorage, SentimentResultModel, ArticleModel
from scraper.news_fetcher import NewsFetcher, _deduplicate, Article
from scraper.rss_fetcher import RSSFetcher
from sentiment.analyzer import SentimentAnalyzer, SentimentResult
from sentiment.scoring import MarketMoodScorer
from dashboard.report import ReportGenerator
from mlops.wandb_tracker import SentimentRunTracker
from mlops.drift_monitor import DriftMonitor
from mlops.model_registry import ModelRegistry
from rag.vector_store import ArticleVectorStore
from rag.rag_chain import MarketInsightRAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_run_lock = threading.Lock()
_pipeline_running = False
_last_run_time = None
_cached_results = []

# research-grade singletons (lazy-init to keep startup fast)
_vector_store = None
_rag_chain = None
_drift_monitor = DriftMonitor()


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        try:
            _vector_store = ArticleVectorStore(persist_dir=config.CHROMA_PERSIST_DIR)
        except Exception as e:
            logger.warning("Vector store init failed: %s", e)
    return _vector_store


def _get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        vs = _get_vector_store()
        if vs and vs.count() > 0:
            try:
                retriever = vs.as_retriever(k=config.RAG_TOP_K)
                _rag_chain = MarketInsightRAG(retriever=retriever)
                logger.info("RAG chain ready (vector store: %d docs)", vs.count())
            except Exception as e:
                logger.warning("RAG chain init failed: %s", e)
    return _rag_chain


def _load_cached_results_from_db():
    global _cached_results
    try:
        from sqlalchemy.orm import Session
        storage = DataStorage()
        with Session(storage.engine) as session:
            rows = (
                session.query(SentimentResultModel)
                .join(ArticleModel)
                .order_by(SentimentResultModel.created_at.desc())
                .limit(50)
                .all()
            )
            rebuilt = []
            for row in rows:
                art = Article(
                    title=row.article.title or "",
                    description=row.article.description or "",
                    url=row.article.url or "",
                    source=row.article.source or "",
                    published_at=row.article.published_at or "",
                    raw_text=f"{row.article.title} {row.article.description}",
                )
                rebuilt.append(SentimentResult(
                    article=art,
                    vader_compound=row.vader_compound or 0.0,
                    vader_label=row.vader_label or "neutral",
                    finbert_label=row.finbert_label or "neutral",
                    finbert_score=row.finbert_score or 0.0,
                    keywords_matched=json.loads(row.keywords_matched) if row.keywords_matched else [],
                    is_safe_haven_signal=bool(row.is_safe_haven_signal),
                    is_risk_on_signal=bool(row.is_risk_on_signal),
                ))
            _cached_results = rebuilt
            logger.info("Warmed headline cache with %d results from DB", len(rebuilt))
    except Exception as e:
        logger.warning("Could not warm cache from DB: %s", e)


def _run_pipeline():
    global _pipeline_running, _last_run_time, _cached_results, _rag_chain

    logger.info("Pipeline starting...")
    tracker = SentimentRunTracker(project=config.WANDB_PROJECT)
    tracker.start_run(config={
        "model": "VADER+FinBERT" if config.USE_FINBERT else "VADER",
        "interval_hours": config.FETCH_INTERVAL_HOURS,
    })

    news = NewsFetcher()
    rss = RSSFetcher()
    articles = _deduplicate(news.fetch_all() + rss.fetch_all_rss())

    if not articles:
        logger.warning("No articles fetched")
        _pipeline_running = False
        tracker.finish()
        return {}

    tracker.log_dataset_stats(articles)
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_batch(articles)

    scorer = MarketMoodScorer()
    mood = scorer.compute_mood_score(results)
    tracker.log_metrics(mood)

    storage = DataStorage()
    url_map = storage.save_articles(articles)
    storage.save_sentiment_results(results, url_map)
    storage.save_mood_snapshot(mood)

    # drift check
    snapshots = storage.get_last_n_snapshots(24)
    if len(snapshots) >= 4:
        drift = _drift_monitor.check_from_snapshots(snapshots)
        mood["drift_report"] = drift

    # update vector store + reset RAG chain
    vs = _get_vector_store()
    if vs:
        vs.add_articles(articles, results)
        _rag_chain = None  # reset so it picks up new docs on next request

    reporter = ReportGenerator()
    os.makedirs("reports", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    with open(f"reports/report_{ts}.json", "w") as f:
        f.write(reporter.generate_json_report(mood, results))
    reporter.generate_html_snapshot(mood, results)

    _cached_results = results
    _last_run_time = datetime.utcnow().isoformat()
    _pipeline_running = False
    tracker.finish()
    logger.info("Pipeline complete — mood: %s", mood.get("mood_label"))
    return mood


# ── routes ───────────────────────────────────────────────────────────────────

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
        snap["headlines"] = _build_headlines(_cached_results)
        snap["last_run_time"] = _last_run_time
        if _cached_results:
            scorer = MarketMoodScorer()
            mood = scorer.compute_mood_score(_cached_results)
            snap["notable_figures"] = mood.get("notable_figures", [])
            snap["positive_count"] = mood.get("positive_count", 0)
            snap["negative_count"] = mood.get("negative_count", 0)
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
        snaps.reverse()
        return jsonify(snaps)
    except Exception as e:
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
    t = threading.Thread(target=_run_pipeline, daemon=True)
    t.start()
    return jsonify({"status": "started", "message": "Analysis started — check back in ~30 seconds"})


@app.route("/api/status")
def api_status():
    vs = _get_vector_store()
    return jsonify({
        "pipeline_running": _pipeline_running,
        "last_run_time": _last_run_time,
        "articles_in_memory": len(_cached_results),
        "vector_store_docs": vs.count() if vs else 0,
        "rag_ready": _get_rag_chain() is not None,
    })


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """RAG-powered market insight endpoint."""
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question field is required"}), 400
    if len(question) > 500:
        return jsonify({"error": "question too long (max 500 chars)"}), 400

    try:
        chain = _get_rag_chain()
        if chain is None:
            # no vector store docs yet — give a helpful fallback
            storage = DataStorage()
            snaps = storage.get_last_n_snapshots(1)
            if snaps:
                snap = snaps[0]
                fallback = (
                    f"Current market mood: {snap.get('mood_label', 'Unknown')} "
                    f"(score: {snap.get('overall_score', 0):+.2f}). "
                    f"Safe-Haven Pressure: {snap.get('shpi', 0):.0%}. "
                    f"Run the analysis pipeline to enable AI-powered insights with source articles."
                )
                return jsonify({"question": question, "answer": fallback, "sources": []})
            return jsonify({"question": question, "answer": "Run the analysis pipeline first to populate the knowledge base.", "sources": []})

        result = chain.ask(question)
        return jsonify(result)
    except Exception as e:
        logger.exception("Error in /api/ask")
        return jsonify({"error": str(e)}), 500


@app.route("/api/search")
def api_search():
    """Semantic similarity search over the article vector store."""
    q = request.args.get("q", "").strip()
    k = request.args.get("k", 5, type=int)
    if not q:
        return jsonify({"error": "q parameter required"}), 400

    vs = _get_vector_store()
    if not vs or vs.count() == 0:
        return jsonify({"results": [], "message": "Vector store empty — run pipeline first"})

    try:
        docs_scores = vs.similarity_search(q, k=k)
        results = []
        for doc, score in docs_scores:
            results.append({
                "title": doc.metadata.get("title", ""),
                "source": doc.metadata.get("source", ""),
                "url": doc.metadata.get("url", ""),
                "sentiment": doc.metadata.get("vader_label", ""),
                "score": doc.metadata.get("vader_compound", 0),
                "similarity": round(float(score), 4),
                "is_safe_haven": doc.metadata.get("is_safe_haven", "false") == "True",
            })
        return jsonify({"query": q, "results": results})
    except Exception as e:
        logger.exception("Error in /api/search")
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-health")
def api_model_health():
    """Returns MLOps health: model registry, drift status, Triton status."""
    registry = ModelRegistry()
    prod_model = registry.get_production("finbert-finetuned")

    storage = DataStorage()
    snapshots = storage.get_last_n_snapshots(24)
    drift_report = {}
    if len(snapshots) >= 4:
        drift_report = _drift_monitor.check_from_snapshots(snapshots)

    vs = _get_vector_store()

    # try to detect Triton status from env
    triton_status = "configured" if config.USE_TRITON else "not_configured"

    return jsonify({
        "sentiment_model": "VADER + FinBERT" if config.USE_FINBERT else "VADER",
        "embedding_model": config.EMBEDDING_MODEL,
        "finbert_backend": "triton" if config.USE_TRITON else "local",
        "triton_status": triton_status,
        "triton_url": config.TRITON_URL if config.USE_TRITON else None,
        "production_model": prod_model,
        "registered_models": registry.all_models(),
        "drift_report": drift_report,
        "vector_store_docs": vs.count() if vs else 0,
        "wandb_project": config.WANDB_PROJECT,
        "rag_ready": _get_rag_chain() is not None,
    })


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_headlines(results, n=7):
    if not results:
        return {"negative": [], "positive": [], "safe_haven": []}

    def fmt(r):
        return {"title": r.article.title, "score": round(r.vader_compound, 3),
                "source": r.article.source, "url": r.article.url}

    return {
        "negative": [fmt(r) for r in sorted(results, key=lambda r: r.vader_compound)[:n]],
        "positive": [fmt(r) for r in sorted(results, key=lambda r: r.vader_compound, reverse=True)[:n]],
        "safe_haven": [fmt(r) for r in results if r.is_safe_haven_signal][:n],
    }


with app.app_context():
    _load_cached_results_from_db()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_run_lock = threading.Lock()
_pipeline_running = False
_last_run_time = None
_cached_results = []


def _load_cached_results_from_db():
    """On startup, warm the in-memory headline cache from the latest DB entries."""
    global _cached_results
    try:
        from sqlalchemy.orm import Session
        storage = DataStorage()
        with Session(storage.engine) as session:
            # get the 50 most recent sentiment results with their articles
            rows = (
                session.query(SentimentResultModel)
                .join(ArticleModel)
                .order_by(SentimentResultModel.created_at.desc())
                .limit(50)
                .all()
            )
            rebuilt = []
            for row in rows:
                art = Article(
                    title=row.article.title or "",
                    description=row.article.description or "",
                    url=row.article.url or "",
                    source=row.article.source or "",
                    published_at=row.article.published_at or "",
                    raw_text=f"{row.article.title} {row.article.description}",
                )
                rebuilt.append(SentimentResult(
                    article=art,
                    vader_compound=row.vader_compound or 0.0,
                    vader_label=row.vader_label or "neutral",
                    finbert_label=row.finbert_label or "neutral",
                    finbert_score=row.finbert_score or 0.0,
                    keywords_matched=json.loads(row.keywords_matched) if row.keywords_matched else [],
                    is_safe_haven_signal=bool(row.is_safe_haven_signal),
                    is_risk_on_signal=bool(row.is_risk_on_signal),
                ))
            _cached_results = rebuilt
            logger.info("Warmed headline cache with %d results from DB", len(rebuilt))
    except Exception as e:
        logger.warning("Could not warm cache from DB: %s", e)


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
        headlines = _build_headlines(_cached_results)
        snap["headlines"] = headlines
        snap["last_run_time"] = _last_run_time
        # include notable figures from the scorer if we have cached results
        if _cached_results:
            scorer = MarketMoodScorer()
            mood = scorer.compute_mood_score(_cached_results)
            snap["notable_figures"] = mood.get("notable_figures", [])
            snap["positive_count"] = mood.get("positive_count", 0)
            snap["negative_count"] = mood.get("negative_count", 0)
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


# warm the cache before the first request
with app.app_context():
    _load_cached_results_from_db()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
