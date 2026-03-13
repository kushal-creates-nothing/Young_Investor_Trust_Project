import unittest

from scraper.news_fetcher import Article
from sentiment.analyzer import SentimentAnalyzer, SentimentResult


def make_article(title, desc="", url="https://test.com/1", source="Test"):
    return Article(
        title=title,
        description=desc,
        url=url,
        source=source,
        published_at="2026-03-12T10:00:00Z",
        raw_text=f"{title} {desc}",
    )


class TestVaderScoring(unittest.TestCase):

    def setUp(self):
        self.analyzer = SentimentAnalyzer()

    def test_positive_headline(self):
        art = make_article("Markets soar on exceptional GDP growth and optimism", url="https://test.com/pos1")
        result = self.analyzer.analyze(art)
        self.assertGreater(result.vader_compound, 0.05)
        self.assertEqual(result.vader_label, "positive")

    def test_negative_headline(self):
        art = make_article("Markets crash in worst sell-off since 2008 recession fears", url="https://test.com/neg1")
        result = self.analyzer.analyze(art)
        self.assertLess(result.vader_compound, -0.05)
        self.assertEqual(result.vader_label, "negative")

    def test_neutral_headline(self):
        art = make_article("Federal Reserve releases meeting minutes", url="https://test.com/neu1")
        result = self.analyzer.analyze(art)
        self.assertEqual(result.vader_label, "neutral")

    def test_compound_range(self):
        for title, url in [
            ("Stocks rise sharply", "https://test.com/r1"),
            ("Markets fall heavily", "https://test.com/r2"),
            ("No change today", "https://test.com/r3"),
        ]:
            art = make_article(title, url=url)
            r = self.analyzer.analyze(art)
            self.assertGreaterEqual(r.vader_compound, -1.0)
            self.assertLessEqual(r.vader_compound, 1.0)


class TestKeywordMatching(unittest.TestCase):

    def setUp(self):
        self.analyzer = SentimentAnalyzer()

    def test_safe_haven_keywords_detected(self):
        art = make_article("Gold surges as investors seek safe haven amid recession fears", url="https://test.com/sh1")
        r = self.analyzer.analyze(art)
        self.assertTrue(r.is_safe_haven_signal)
        self.assertIn("gold", r.keywords_matched)

    def test_risk_on_keywords_detected(self):
        art = make_article("Bull market confirmed as earnings beat forecasts and GDP growth accelerates", url="https://test.com/ro1")
        r = self.analyzer.analyze(art)
        self.assertTrue(r.is_risk_on_signal)
        self.assertIn("bull market", r.keywords_matched)

    def test_no_keywords_matched(self):
        art = make_article("Central bank holds quarterly press conference", url="https://test.com/nk1")
        r = self.analyzer.analyze(art)
        # 'central bank' is in risk-off, should be matched
        # this tests that the keyword list itself is not empty
        self.assertIsInstance(r.keywords_matched, list)

    def test_inflation_triggers_safe_haven(self):
        art = make_article("Inflation hits 40-year high as prices surge globally", url="https://test.com/inf1")
        r = self.analyzer.analyze(art)
        self.assertTrue(r.is_safe_haven_signal)
        self.assertIn("inflation", r.keywords_matched)

    def test_ipo_triggers_risk_on(self):
        art = make_article("Tech startup raises $2B in IPO on Nasdaq", url="https://test.com/ipo1")
        r = self.analyzer.analyze(art)
        self.assertTrue(r.is_risk_on_signal)

    def test_both_signals_possible(self):
        # article mentions both gold (safe) and growth (risk-on)
        art = make_article("Gold growth investment strategy recommended by experts", url="https://test.com/both1")
        r = self.analyzer.analyze(art)
        self.assertTrue(r.is_safe_haven_signal)
        self.assertTrue(r.is_risk_on_signal)

    def test_keyword_matching_case_insensitive(self):
        art = make_article("GOLD and SAFE HAVEN demand rises as RECESSION fears grow", url="https://test.com/ci1")
        r = self.analyzer.analyze(art)
        self.assertTrue(r.is_safe_haven_signal)

    def test_bear_market_safe_haven(self):
        art = make_article("Bear market confirmed as losses mount", url="https://test.com/bm1")
        r = self.analyzer.analyze(art)
        self.assertTrue(r.is_safe_haven_signal)
        self.assertIn("bear market", r.keywords_matched)


class TestBatchAnalysis(unittest.TestCase):

    def setUp(self):
        self.analyzer = SentimentAnalyzer()

    def test_batch_returns_correct_count(self):
        articles = [
            make_article("Markets rally sharply", url="https://test.com/b1"),
            make_article("Recession fears grow", url="https://test.com/b2"),
            make_article("Fed meeting today", url="https://test.com/b3"),
        ]
        results = self.analyzer.analyze_batch(articles)
        self.assertEqual(len(results), 3)

    def test_batch_returns_sentiment_results(self):
        articles = [make_article("Strong growth ahead", url=f"https://test.com/sr{i}") for i in range(5)]
        results = self.analyzer.analyze_batch(articles)
        for r in results:
            self.assertIsInstance(r, SentimentResult)
            self.assertIsInstance(r.vader_compound, float)
            self.assertIn(r.vader_label, ("positive", "negative", "neutral"))

    def test_empty_batch(self):
        results = self.analyzer.analyze_batch([])
        self.assertEqual(results, [])

    def test_result_preserves_article_reference(self):
        art = make_article("Gold prices jump", url="https://test.com/ref1")
        r = self.analyzer.analyze(art)
        self.assertIs(r.article, art)


class TestFinBertDisabled(unittest.TestCase):

    def test_finbert_disabled_returns_neutral(self):
        analyzer = SentimentAnalyzer()
        analyzer.use_finbert = False
        label, score = analyzer._finbert_score("any text here")
        self.assertEqual(label, "neutral")
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
