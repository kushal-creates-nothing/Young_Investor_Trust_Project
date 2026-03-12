import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# W&B is optional — fall back to offline/noop mode if not configured
try:
    import wandb
    _HAVE_WANDB = True
except ImportError:
    _HAVE_WANDB = False
    logger.warning("wandb not installed — experiment tracking disabled")


class SentimentRunTracker:
    """
    Weights & Biases experiment tracker for sentiment analysis pipeline runs.

    Tracks per-run metrics (mood score, SHPI, fear score, article count),
    model information, and dataset statistics. Falls back silently to a
    no-op tracker when no W&B API key is available.

    Usage:
        tracker = SentimentRunTracker()
        tracker.start_run(config={"model": "VADER", "interval_hours": 2})
        tracker.log_metrics(mood_dict)
        tracker.log_dataset_stats(articles)
        tracker.finish()
    """

    PROJECT = "young-investor-trust"

    def __init__(self, project: str = None, entity: str = None):
        self._project = project or self.PROJECT
        self._entity = entity or os.getenv("WANDB_ENTITY", None)
        self._run = None
        self._active = False
        self._run_metrics = []  # local buffer when W&B is offline

    def start_run(self, config: dict = None, run_name: str = None) -> bool:
        if not _HAVE_WANDB:
            return False

        api_key = os.getenv("WANDB_API_KEY", "")
        mode = "online" if api_key else "offline"

        try:
            self._run = wandb.init(
                project=self._project,
                entity=self._entity,
                name=run_name or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                config=config or {},
                mode=mode,
                reinit="return_previous",
            )
            self._active = True
            logger.info("W&B run started (mode=%s): %s", mode, getattr(self._run, "name", "unknown"))
            return True
        except Exception as e:
            logger.warning("W&B init failed: %s — continuing without tracking", e)
            return False

    def log_metrics(self, mood: dict, step: int = None):
        metrics = {
            "overall_sentiment": mood.get("overall_score", 0),
            "shpi": mood.get("shpi", 0),
            "risk_on_index": mood.get("risk_on_index", 0),
            "fear_score": mood.get("fear_score", 0),
            "article_count": mood.get("article_count", 0),
            "safe_haven_count": mood.get("safe_haven_count", 0),
            "tipping_point_active": 1 if mood.get("tipping_point_alert") else 0,
        }
        self._run_metrics.append({**metrics, "timestamp": datetime.utcnow().isoformat()})

        if self._active and self._run:
            try:
                log_kwargs = {"step": step} if step is not None else {}
                self._run.log(metrics, **log_kwargs)
            except Exception as e:
                logger.warning("W&B log_metrics failed: %s", e)

    def log_dataset_stats(self, articles):
        stats = {
            "total_articles": len(articles),
            "sources": len({a.source for a in articles}),
        }
        if self._active and self._run:
            try:
                self._run.log(stats)
            except Exception as e:
                logger.warning("W&B log_dataset_stats failed: %s", e)

    def log_model_info(self, model_name: str, model_path: str = None, metrics: dict = None):
        if self._active and self._run:
            try:
                self._run.config.update({"model_name": model_name})
                if metrics:
                    self._run.log({f"model/{k}": v for k, v in metrics.items()})
                if model_path and os.path.exists(model_path):
                    artifact = wandb.Artifact(name="finbert-finetuned", type="model")
                    artifact.add_dir(model_path)
                    self._run.log_artifact(artifact)
                    logger.info("Model artifact logged to W&B: %s", model_path)
            except Exception as e:
                logger.warning("W&B log_model_info failed: %s", e)

    def finish(self):
        if self._active and self._run:
            try:
                self._run.finish()
                logger.info("W&B run finished: %s", getattr(self._run, "url", "offline"))
            except Exception as e:
                logger.warning("W&B finish failed: %s", e)
        self._active = False
        self._run = None

    @property
    def run_url(self) -> str:
        if self._run and hasattr(self._run, "url"):
            return self._run.url or ""
        return ""

    @property
    def local_metrics(self) -> list:
        return self._run_metrics.copy()
