import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

REGISTRY_PATH = os.getenv("MODEL_REGISTRY_PATH", "data/model_registry.json")


class ModelRegistry:
    """
    Lightweight local model registry that tracks fine-tuned model versions,
    evaluation metrics, and deployment status.

    In production you'd swap this for MLflow Model Registry or W&B Artifacts,
    but this file-backed version works offline and is easy to inspect manually.
    """

    def __init__(self, registry_path: str = None):
        self._path = registry_path or REGISTRY_PATH
        self._data = self._load()

    def register(
        self,
        model_name: str,
        model_path: str,
        metrics: dict,
        base_model: str = "ProsusAI/finbert",
        tags: list = None,
    ) -> str:
        """Register a new model version and return its version string."""
        versions = self._data.get(model_name, [])
        version = f"v{len(versions) + 1}.0"

        entry = {
            "version": version,
            "model_path": model_path,
            "base_model": base_model,
            "registered_at": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "tags": tags or [],
            "status": "staging",  # staging → production → archived
        }
        versions.append(entry)
        self._data[model_name] = versions
        self._save()
        logger.info("Registered %s %s — metrics: %s", model_name, version, metrics)
        return version

    def promote(self, model_name: str, version: str) -> bool:
        """Promote a model version to production (demotes the current one)."""
        versions = self._data.get(model_name, [])
        promoted = False
        for entry in versions:
            if entry["version"] == version:
                entry["status"] = "production"
                promoted = True
            elif entry["status"] == "production":
                entry["status"] = "archived"
        if promoted:
            self._save()
            logger.info("Promoted %s %s to production", model_name, version)
        return promoted

    def get_production(self, model_name: str) -> Optional[dict]:
        """Return the current production version of a model, or None."""
        for entry in reversed(self._data.get(model_name, [])):
            if entry["status"] == "production":
                return entry
        return None

    def list_versions(self, model_name: str) -> list:
        return self._data.get(model_name, [])

    def all_models(self) -> list:
        return list(self._data.keys())

    # ── persistence ────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not load registry at '%s': %s", self._path, e)
        return {}

    def _save(self):
        os.makedirs(os.path.dirname(self._path) if os.path.dirname(self._path) else ".", exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)
