import unittest

from scraper.news_fetcher import Article
from sentiment.analyzer import SentimentResult
from sentiment.scoring import MarketMoodScorer


def make_result(title, compound, is_safe=False, is_risk=False, url_suffix="1"):
    label = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")
    art = Article(
        title=title,
        description="",
        url=f"https://test.com/{url_suffix}",
        source="Test",
        published_at="2026-03-12T00:00:00Z",
        raw_text=title,
    )
    return SentimentResult(
        article=art,
        vader_compound=compound,
        vader_label=label,
        finbert_label=label,
        finbert_score=abs(compound),
        keywords_matched=[],
        is_safe_haven_signal=is_safe,
        is_risk_on_signal=is_risk,
    )


class TestComputeMoodScore(unittest.TestCase):

    def setUp(self):
        self.scorer = MarketMoodScorer()

    def test_empty_returns_neutral(self):
        mood = self.scorer.compute_mood_score([])
        self.assertEqual(mood["mood_label"], "Neutral")
        self.assertEqual(mood["overall_score"], 0.0)
        self.assertEqual(mood["article_count"], 0)

    def test_strongly_bullish(self):
        results = [make_result("Markets rally hard", 0.8, url_suffix=str(i)) for i in range(5)]
        mood = self.scorer.compute_mood_score(results)
        self.assertEqual(mood["mood_label"], "Strongly Bullish")
        self.assertGreater(mood["overall_score"], 0.35)

    def test_bullish(self):
        results = [make_result("Good economic news", 0.2, url_suffix=str(i)) for i in range(5)]
        mood = self.scorer.compute_mood_score(results)
        self.assertEqual(mood["mood_label"], "Bullish")

    def test_neutral(self):
        results = [make_result("Markets unchanged", 0.0, url_suffix=str(i)) for i in range(5)]
        mood = self.scorer.compute_mood_score(results)
        self.assertEqual(mood["mood_label"], "Neutral")

    def test_bearish(self):
        results = [make_result("Markets fall slightly", -0.25, url_suffix=str(i)) for i in range(5)]
        mood = self.scorer.compute_mood_score(results)
        self.assertEqual(mood["mood_label"], "Bearish")

    def test_strongly_bearish(self):
        results = [make_result("Catastrophic crash", -0.8, url_suffix=str(i)) for i in range(5)]
        mood = self.scorer.compute_mood_score(results)
        self.assertEqual(mood["mood_label"], "Strongly Bearish")

    def test_safe_haven_flight_label(self):
        # SHPI = 8/10 = 0.8 (> 0.4), overall = (8*-0.4 + 2*0.4)/10 = -0.24 (< -0.2)
        results = (
            [make_result("Gold safe haven bonds", -0.4, is_safe=True, url_suffix=str(i)) for i in range(8)]
            + [make_result("Markets rally", 0.4, url_suffix=str(i+20)) for i in range(2)]
        )
        mood = self.scorer.compute_mood_score(results)
        self.assertEqual(mood["mood_label"], "Safe-Haven Flight")


class TestSHPICalculation(unittest.TestCase):

    def setUp(self):
        self.scorer = MarketMoodScorer()

    def test_shpi_zero_when_no_safe_haven(self):
        results = [make_result("Markets rally", 0.3, url_suffix=str(i)) for i in range(5)]
        mood = self.scorer.compute_mood_score(results)
        self.assertEqual(mood["shpi"], 0.0)

    def test_shpi_one_when_all_safe_haven(self):
        results = [make_result("Gold", 0.0, is_safe=True, url_suffix=str(i)) for i in range(5)]
        mood = self.scorer.compute_mood_score(results)
        self.assertEqual(mood["shpi"], 1.0)

    def test_shpi_partial(self):
        results = (
            [make_result("Gold bonds", -0.2, is_safe=True, url_suffix=str(i)) for i in range(4)]
            + [make_result("Tech rally", 0.5, url_suffix=str(i+10)) for i in range(6)]
        )
        mood = self.scorer.compute_mood_score(results)
        self.assertAlmostEqual(mood["shpi"], 0.4, places=2)


class TestTippingPointAlert(unittest.TestCase):

    def setUp(self):
        self.scorer = MarketMoodScorer()

    def test_tipping_point_fires(self):
        # SHPI > 0.4 AND overall < -0.2
        results = [make_result("Safe haven gold bonds", -0.4, is_safe=True, url_suffix=str(i)) for i in range(10)]
        mood = self.scorer.compute_mood_score(results)
        self.assertIn("CAUTION", mood["tipping_point_alert"])

    def test_approaching_alert_fires(self):
        # SHPI > 0.3 AND overall < 0
        results = (
            [make_result("Gold safe haven", -0.15, is_safe=True, url_suffix=str(i)) for i in range(4)]
            + [make_result("Markets flat", -0.05, url_suffix=str(i+10)) for i in range(6)]
        )
        mood = self.scorer.compute_mood_score(results)
        self.assertTrue(len(mood["tipping_point_alert"]) > 0)

    def test_no_alert_when_bullish(self):
        results = [make_result("Bull market rally", 0.6, url_suffix=str(i)) for i in range(10)]
        mood = self.scorer.compute_mood_score(results)
        self.assertEqual(mood["tipping_point_alert"], "")

    def test_no_alert_when_shpi_low(self):
        results = (
            [make_result("Safe gold", -0.3, is_safe=True, url_suffix=str(i)) for i in range(2)]
            + [make_result("Markets", 0.0, url_suffix=str(i+10)) for i in range(8)]
        )
        mood = self.scorer.compute_mood_score(results)
        self.assertEqual(mood["tipping_point_alert"], "")


class TestFearScore(unittest.TestCase):

    def setUp(self):
        self.scorer = MarketMoodScorer()

    def test_fear_increases_with_negative_articles(self):
        low_fear = [make_result("Great news", 0.7, url_suffix=str(i)) for i in range(10)]
        high_fear = [make_result("Crash recession", -0.7, is_safe=True, url_suffix=str(i+20)) for i in range(10)]
        m_low = self.scorer.compute_mood_score(low_fear)
        m_high = self.scorer.compute_mood_score(high_fear)
        self.assertGreater(m_high["fear_score"], m_low["fear_score"])

    def test_fear_capped_at_one(self):
        results = [make_result("Crash gold safe", -0.9, is_safe=True, url_suffix=str(i)) for i in range(20)]
        mood = self.scorer.compute_mood_score(results)
        self.assertLessEqual(mood["fear_score"], 1.0)


class TestNotableFigures(unittest.TestCase):

    def setUp(self):
        self.scorer = MarketMoodScorer()

    def test_detects_jerome_powell(self):
        arts = [
            make_result("Jerome Powell warns of more rate hikes ahead", -0.55, url_suffix="jpa"),
            make_result("Jerome Powell signals caution on rate cuts", -0.41, url_suffix="jpb"),
        ]
        # inject raw_text so the name is found
        for r in arts:
            r.article.raw_text = r.article.title

        mood = self.scorer.compute_mood_score(arts)
        names = [f["name"] for f in mood["notable_figures"]]
        self.assertIn("Jerome Powell", names)
        jp = next(f for f in mood["notable_figures"] if f["name"] == "Jerome Powell")
        self.assertEqual(jp["sentiment_label"], "negative")

    def test_no_figures_when_none_mentioned(self):
        results = [make_result("Markets fall today", -0.3, url_suffix=str(i)) for i in range(5)]
        mood = self.scorer.compute_mood_score(results)
        self.assertEqual(mood["notable_figures"], [])


if __name__ == "__main__":
    unittest.main()
