import unittest
from unittest.mock import MagicMock, patch
import numpy as np


class TestDriftMonitor(unittest.TestCase):

    def setUp(self):
        from mlops.drift_monitor import DriftMonitor
        self.monitor = DriftMonitor()

    def test_identical_distributions_no_drift(self):
        # Use large seeded samples from the same distribution
        rng = np.random.default_rng(123)
        baseline = list(rng.normal(0.1, 0.3, 200))
        current = list(rng.normal(0.1, 0.3, 200))
        result = self.monitor.check_sentiment_drift(baseline, current)
        # same distribution: KS p-value should be > 0.05
        self.assertGreater(result["ks_p_value"], 0.05)

    def test_psi_stable_distribution(self):
        # PSI between two large samples from the same distribution should be < 0.5
        rng = np.random.default_rng(7)
        baseline = list(rng.normal(0.0, 0.2, 200))
        current = list(rng.normal(0.0, 0.2, 200))
        result = self.monitor.check_sentiment_drift(baseline, current)
        self.assertLess(result["psi"], 0.5)
        self.assertIn(result["severity"], ("ok", "insufficient_data"))

    def test_very_different_distributions_drift(self):
        baseline = list(np.random.normal(0.3, 0.1, 100))   # bullish baseline
        current = list(np.random.normal(-0.5, 0.1, 60))    # suddenly bearish
        result = self.monitor.check_sentiment_drift(baseline, current)
        self.assertTrue(result["drifted"])
        self.assertIn(result["severity"], ("warning", "critical"))

    def test_insufficient_data_returns_safe(self):
        result = self.monitor.check_sentiment_drift([0.1, 0.2], [0.3])
        self.assertFalse(result["drifted"])
        self.assertEqual(result["severity"], "insufficient_data")

    def test_empty_inputs(self):
        result = self.monitor.check_sentiment_drift([], [])
        self.assertFalse(result["drifted"])

    def test_result_contains_all_keys(self):
        bl = list(np.random.normal(0.0, 0.2, 50))
        cur = list(np.random.normal(-0.2, 0.2, 30))
        result = self.monitor.check_sentiment_drift(bl, cur)
        for key in ("drifted", "ks_statistic", "ks_p_value", "psi", "severity", "recommendation"):
            self.assertIn(key, result)

    def test_mean_shift_computed_correctly(self):
        bl = [0.3] * 50
        cur = [-0.3] * 30
        result = self.monitor.check_sentiment_drift(bl, cur)
        self.assertAlmostEqual(result["baseline_mean"], 0.3, places=3)
        self.assertAlmostEqual(result["current_mean"], -0.3, places=3)
        self.assertAlmostEqual(result["mean_shift"], -0.6, places=3)

    def test_check_from_snapshots_too_few(self):
        result = self.monitor.check_from_snapshots([{"overall_score": 0.1}])
        self.assertFalse(result["drifted"])
        self.assertEqual(result["severity"], "insufficient_data")

    def test_check_from_snapshots_stable(self):
        snaps = [{"overall_score": 0.1, "shpi": 0.2} for _ in range(12)]
        result = self.monitor.check_from_snapshots(snaps)
        self.assertIn("sentiment_drift", result)
        self.assertIn("shpi_drift", result)
        self.assertIn("overall_status", result)


class TestWandbTracker(unittest.TestCase):

    def setUp(self):
        from mlops.wandb_tracker import SentimentRunTracker
        self.tracker = SentimentRunTracker(project="test-project")

    @patch("mlops.wandb_tracker._HAVE_WANDB", False)
    def test_start_run_without_wandb(self):
        result = self.tracker.start_run()
        self.assertFalse(result)

    def test_log_metrics_stores_locally(self):
        mood = {
            "overall_score": -0.31,
            "shpi": 0.42,
            "risk_on_index": 0.14,
            "fear_score": 0.62,
            "article_count": 20,
            "safe_haven_count": 8,
            "tipping_point_alert": "⚠️ CAUTION",
        }
        self.tracker.log_metrics(mood)
        self.assertEqual(len(self.tracker.local_metrics), 1)
        stored = self.tracker.local_metrics[0]
        self.assertAlmostEqual(stored["overall_sentiment"], -0.31)
        self.assertAlmostEqual(stored["shpi"], 0.42)
        self.assertEqual(stored["tipping_point_active"], 1)

    def test_log_metrics_tipping_point_inactive(self):
        mood = {"overall_score": 0.3, "shpi": 0.1, "risk_on_index": 0.4,
                "fear_score": 0.2, "article_count": 15, "safe_haven_count": 2,
                "tipping_point_alert": ""}
        self.tracker.log_metrics(mood)
        stored = self.tracker.local_metrics[0]
        self.assertEqual(stored["tipping_point_active"], 0)

    def test_finish_does_not_crash_when_not_started(self):
        self.tracker.finish()  # should not raise

    def test_run_url_empty_when_not_started(self):
        self.assertEqual(self.tracker.run_url, "")

    def test_multiple_metrics_accumulate(self):
        mood = {"overall_score": 0.1, "shpi": 0.2, "risk_on_index": 0.3,
                "fear_score": 0.4, "article_count": 10, "safe_haven_count": 2,
                "tipping_point_alert": ""}
        self.tracker.log_metrics(mood)
        self.tracker.log_metrics(mood)
        self.tracker.log_metrics(mood)
        self.assertEqual(len(self.tracker.local_metrics), 3)


class TestModelRegistry(unittest.TestCase):

    def setUp(self):
        import tempfile
        import os
        self.tmp = tempfile.mktemp(suffix=".json")
        from mlops.model_registry import ModelRegistry
        self.registry = ModelRegistry(registry_path=self.tmp)

    def tearDown(self):
        import os
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_register_creates_entry(self):
        version = self.registry.register(
            model_name="finbert-test",
            model_path="/tmp/model",
            metrics={"accuracy": 0.87, "f1_weighted": 0.85},
            base_model="ProsusAI/finbert",
        )
        self.assertEqual(version, "v1.0")
        versions = self.registry.list_versions("finbert-test")
        self.assertEqual(len(versions), 1)

    def test_multiple_versions_increment(self):
        self.registry.register("finbert-test", "/tmp/v1", {"accuracy": 0.85})
        self.registry.register("finbert-test", "/tmp/v2", {"accuracy": 0.87})
        versions = self.registry.list_versions("finbert-test")
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[0]["version"], "v1.0")
        self.assertEqual(versions[1]["version"], "v2.0")

    def test_promote_sets_production(self):
        self.registry.register("finbert-test", "/tmp/v1", {"accuracy": 0.85})
        self.registry.promote("finbert-test", "v1.0")
        prod = self.registry.get_production("finbert-test")
        self.assertIsNotNone(prod)
        self.assertEqual(prod["version"], "v1.0")
        self.assertEqual(prod["status"], "production")

    def test_promote_demotes_old_production(self):
        self.registry.register("finbert-test", "/tmp/v1", {"accuracy": 0.85})
        self.registry.register("finbert-test", "/tmp/v2", {"accuracy": 0.88})
        self.registry.promote("finbert-test", "v1.0")
        self.registry.promote("finbert-test", "v2.0")
        versions = self.registry.list_versions("finbert-test")
        v1 = next(v for v in versions if v["version"] == "v1.0")
        v2 = next(v for v in versions if v["version"] == "v2.0")
        self.assertEqual(v1["status"], "archived")
        self.assertEqual(v2["status"], "production")

    def test_no_production_returns_none(self):
        self.registry.register("finbert-test", "/tmp/v1", {})
        prod = self.registry.get_production("finbert-test")
        self.assertIsNone(prod)

    def test_all_models_lists_registered(self):
        self.registry.register("model-a", "/tmp/a", {})
        self.registry.register("model-b", "/tmp/b", {})
        models = self.registry.all_models()
        self.assertIn("model-a", models)
        self.assertIn("model-b", models)

    def test_persistence_across_instances(self):
        self.registry.register("finbert-test", "/tmp/v1", {"accuracy": 0.9})
        from mlops.model_registry import ModelRegistry
        registry2 = ModelRegistry(registry_path=self.tmp)
        versions = registry2.list_versions("finbert-test")
        self.assertEqual(len(versions), 1)

    def test_promote_nonexistent_returns_false(self):
        result = self.registry.promote("no-such-model", "v1.0")
        self.assertFalse(result)


class TestTritonClient(unittest.TestCase):

    @patch("serving.triton_client._HAVE_TRITON", False)
    def test_classify_without_triton_returns_tuple(self):
        from serving.triton_client import TritonSentimentClient
        client = TritonSentimentClient.__new__(TritonSentimentClient)
        client._using_triton = False
        client._fallback = None
        label, score = client._local_classify("Gold prices surge amid fear")
        self.assertEqual(label, "neutral")
        self.assertEqual(score, 0.0)

    def test_labels_constant(self):
        from serving.triton_client import TritonSentimentClient
        self.assertEqual(TritonSentimentClient.LABELS, ["positive", "negative", "neutral"])

    def test_classify_batch_returns_list(self):
        from serving.triton_client import TritonSentimentClient
        client = TritonSentimentClient.__new__(TritonSentimentClient)
        client._using_triton = False
        client._fallback = None
        results = client.classify_batch(["text 1", "text 2", "text 3"])
        self.assertEqual(len(results), 3)
        for label, score in results:
            self.assertIsInstance(label, str)
            self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()
