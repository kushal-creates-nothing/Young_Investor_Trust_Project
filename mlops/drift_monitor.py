import logging
import math
from typing import List, Optional

from scipy import stats
import numpy as np

logger = logging.getLogger(__name__)

# thresholds — these were tuned empirically; adjust based on your baseline data
KS_ALPHA = 0.05        # p-value threshold for KS test
PSI_WARNING = 0.1      # Population Stability Index: slight change
PSI_CRITICAL = 0.2     # PSI: significant change (retraining needed)


class DriftMonitor:
    """
    Detects statistical drift in sentiment distributions between pipeline runs.

    Uses two complementary tests:
      1. Kolmogorov-Smirnov (KS) test — detects distributional shift
      2. Population Stability Index (PSI) — standard ML drift metric

    In practice, after enough pipeline runs accumulate in the DB, you'd compare
    the last N runs against a stable baseline window. The monitor flags when
    the distribution of VADER compound scores has shifted significantly —
    which could indicate either:
      a) actual change in market sentiment (expected, fine)
      b) model degradation or data pipeline issues (needs investigation)
    """

    def __init__(self, baseline_window: int = 100, current_window: int = 50):
        self.baseline_window = baseline_window
        self.current_window = current_window

    def check_sentiment_drift(
        self,
        baseline_scores: List[float],
        current_scores: List[float],
    ) -> dict:
        """
        Run KS test + PSI on two distributions of VADER compound scores.
        Returns a dict with drift status, test statistics, and an action recommendation.
        """
        if len(baseline_scores) < 10 or len(current_scores) < 5:
            return {
                "drifted": False,
                "ks_statistic": None,
                "ks_p_value": None,
                "psi": None,
                "severity": "insufficient_data",
                "recommendation": "Need more data to assess drift",
            }

        bl = np.array(baseline_scores, dtype=float)
        cur = np.array(current_scores, dtype=float)

        # KS test
        ks_stat, p_value = stats.ks_2samp(bl, cur)
        ks_drifted = p_value < KS_ALPHA

        # PSI
        psi = self._compute_psi(bl, cur)
        if psi >= PSI_CRITICAL:
            severity = "critical"
            action = "Significant distribution shift detected — consider retraining FinBERT or checking data pipeline"
        elif psi >= PSI_WARNING:
            severity = "warning"
            action = "Moderate shift detected — monitor closely over next few runs"
        else:
            severity = "ok"
            action = "Distribution stable — no action needed"

        drifted = ks_drifted or psi >= PSI_WARNING

        return {
            "drifted": drifted,
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": round(float(p_value), 4),
            "psi": round(float(psi), 4),
            "severity": severity,
            "recommendation": action,
            "baseline_mean": round(float(bl.mean()), 4),
            "current_mean": round(float(cur.mean()), 4),
            "mean_shift": round(float(cur.mean() - bl.mean()), 4),
        }

    def check_shpi_drift(
        self,
        baseline_shpi: List[float],
        current_shpi: List[float],
    ) -> dict:
        """Same drift check but for SHPI values across runs."""
        return self.check_sentiment_drift(baseline_shpi, current_shpi)

    def check_from_snapshots(self, snapshots: List[dict]) -> dict:
        """
        Convenience method: splits a list of mood snapshots into a baseline
        (older half) and current (newer half) and runs drift detection.
        """
        if len(snapshots) < 4:
            return {"drifted": False, "severity": "insufficient_data", "recommendation": "Need at least 4 snapshots"}

        mid = len(snapshots) // 2
        baseline_scores = [s["overall_score"] for s in snapshots[:mid] if s.get("overall_score") is not None]
        current_scores = [s["overall_score"] for s in snapshots[mid:] if s.get("overall_score") is not None]

        sentiment_drift = self.check_sentiment_drift(baseline_scores, current_scores)

        baseline_shpi = [s["shpi"] for s in snapshots[:mid] if s.get("shpi") is not None]
        current_shpi = [s["shpi"] for s in snapshots[mid:] if s.get("shpi") is not None]
        shpi_drift = self.check_shpi_drift(baseline_shpi, current_shpi)

        return {
            "sentiment_drift": sentiment_drift,
            "shpi_drift": shpi_drift,
            "overall_status": "warning" if (sentiment_drift["drifted"] or shpi_drift["drifted"]) else "ok",
        }

    @staticmethod
    def _compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        """
        Population Stability Index — industry standard for model drift.
        PSI < 0.1: stable, 0.1-0.2: slight change, >0.2: significant change.
        """
        # bin edges across the combined range
        min_val = min(expected.min(), actual.min()) - 1e-9
        max_val = max(expected.max(), actual.max()) + 1e-9
        bins = np.linspace(min_val, max_val, n_bins + 1)

        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        # convert to proportions, avoid zero-division
        exp_pct = (expected_counts / len(expected)).clip(1e-6)
        act_pct = (actual_counts / len(actual)).clip(1e-6)

        psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
        return psi
