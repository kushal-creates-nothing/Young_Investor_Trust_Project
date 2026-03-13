import unittest
from unittest.mock import patch, MagicMock
import json
import requests

from scraper.news_fetcher import NewsFetcher, Article, _deduplicate
from scraper.rss_fetcher import RSSFetcher


NEWSAPI_RESPONSE = {
    "articles": [
        {
            "title": "Markets rally on strong earnings",
            "description": "US stocks climbed after better-than-expected results.",
            "url": "https://example.com/1",
            "source": {"name": "Test Source"},
            "publishedAt": "2026-03-12T10:00:00Z",
        },
        {
            "title": "Gold surges amid recession fears",
            "description": "Safe-haven demand pushes gold to new highs.",
            "url": "https://example.com/2",
            "source": {"name": "Test Source"},
            "publishedAt": "2026-03-12T09:00:00Z",
        },
    ]
}

GNEWS_RESPONSE = {
    "articles": [
        {
            "title": "Fed holds rates steady",
            "description": "Federal Reserve pauses rate hikes.",
            "url": "https://gnews.example.com/1",
            "source": {"name": "GNews Source"},
            "publishedAt": "2026-03-12T08:00:00Z",
        }
    ]
}

FINNHUB_RESPONSE = [
    {
        "headline": "IPO boom continues in tech sector",
        "summary": "Several major tech companies are planning IPOs this year.",
        "url": "https://finnhub.example.com/1",
        "source": "Finnhub",
        "datetime": 1741776000,
    }
]

SAMPLE_RSS_XML = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Bear market territory reached</title>
      <description>Stocks fall 20% from peak.</description>
      <link>https://rss.example.com/1</link>
      <pubDate>Thu, 12 Mar 2026 10:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Central bank raises interest rates</title>
      <description>Rate hike surprise shocks markets.</description>
      <link>https://rss.example.com/2</link>
      <pubDate>Thu, 12 Mar 2026 09:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""


class TestNewsFetcher(unittest.TestCase):

    def setUp(self):
        self.fetcher = NewsFetcher()
        self.fetcher.newsapi_key = "fake_key"
        self.fetcher.gnews_key = "fake_key"
        self.fetcher.finnhub_key = "fake_key"

    @patch("scraper.news_fetcher.requests.get")
    def test_fetch_newsapi_returns_articles(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = NEWSAPI_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        articles = self.fetcher.fetch_newsapi()
        self.assertEqual(len(articles), 2)
        self.assertIsInstance(articles[0], Article)
        self.assertEqual(articles[0].title, "Markets rally on strong earnings")
        self.assertEqual(articles[0].source, "Test Source")

    @patch("scraper.news_fetcher.requests.get")
    def test_fetch_newsapi_empty_on_failure(self, mock_get):
        mock_get.side_effect = requests.RequestException("network error")
        articles = self.fetcher.fetch_newsapi()
        self.assertEqual(articles, [])

    @patch("scraper.news_fetcher.requests.get")
    def test_fetch_gnews_returns_articles(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = GNEWS_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        articles = self.fetcher.fetch_gnews()
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "Fed holds rates steady")

    @patch("scraper.news_fetcher.requests.get")
    def test_fetch_finnhub_returns_articles(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = FINNHUB_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        articles = self.fetcher.fetch_finnhub()
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "IPO boom continues in tech sector")

    def test_fetch_newsapi_skipped_when_no_key(self):
        self.fetcher.newsapi_key = ""
        articles = self.fetcher.fetch_newsapi()
        self.assertEqual(articles, [])

    def test_deduplication_by_url(self):
        a1 = Article("Title A", "Desc", "https://ex.com/1", "Source", "2026-01-01")
        a2 = Article("Title B", "Desc", "https://ex.com/1", "Source", "2026-01-01")
        a3 = Article("Title C", "Desc", "https://ex.com/2", "Source", "2026-01-01")
        result = _deduplicate([a1, a2, a3])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].title, "Title A")

    def test_deduplication_by_title(self):
        a1 = Article("Same Title", "Desc", "https://ex.com/1", "Source", "2026-01-01")
        a2 = Article("Same Title", "Desc", "https://ex.com/2", "Source", "2026-01-01")
        result = _deduplicate([a1, a2])
        self.assertEqual(len(result), 1)

    def test_deduplication_case_insensitive(self):
        a1 = Article("Gold Hits Record High", "Desc", "https://ex.com/1", "Source", "2026-01-01")
        a2 = Article("gold hits record high", "Desc", "https://ex.com/2", "Source", "2026-01-01")
        result = _deduplicate([a1, a2])
        self.assertEqual(len(result), 1)

    @patch("scraper.rss_fetcher.feedparser.parse")
    def test_rss_fetcher_parses_feed(self, mock_parse):
        # Build a mock feedparser result directly rather than parsing XML
        # (calling feedparser.parse inside the test would hit the mock itself)
        entry1 = MagicMock()
        entry1.get = lambda k, d="": {
            "title": "Bear market territory reached",
            "summary": "Stocks fall 20% from peak.",
            "link": "https://rss.example.com/1",
            "published": "Thu, 12 Mar 2026 10:00:00 +0000",
        }.get(k, d)

        entry2 = MagicMock()
        entry2.get = lambda k, d="": {
            "title": "Central bank raises interest rates",
            "summary": "Rate hike surprise shocks markets.",
            "link": "https://rss.example.com/2",
            "published": "Thu, 12 Mar 2026 09:00:00 +0000",
        }.get(k, d)

        mock_feed = MagicMock()
        mock_feed.entries = [entry1, entry2]
        mock_parse.return_value = mock_feed

        fetcher = RSSFetcher(feeds={"Test Feed": "http://fake.rss/feed"})
        articles = fetcher.fetch_all_rss()
        self.assertEqual(len(articles), 2)
        self.assertIn("Bear market", articles[0].title)

    @patch("scraper.rss_fetcher.feedparser.parse")
    def test_rss_fetcher_handles_error(self, mock_parse):
        mock_parse.side_effect = Exception("connection refused")
        fetcher = RSSFetcher(feeds={"Bad Feed": "http://broken.rss/"})
        articles = fetcher.fetch_all_rss()
        self.assertEqual(articles, [])

    @patch("scraper.news_fetcher.requests.get")
    @patch("scraper.rss_fetcher.feedparser.parse")
    def test_fetch_all_deduplicates_across_sources(self, mock_parse, mock_get):
        # NewsAPI returns article with same title as RSS
        mock_api_resp = MagicMock()
        mock_api_resp.raise_for_status = MagicMock()
        mock_api_resp.json.return_value = {
            "articles": [{
                "title": "Markets rally on strong earnings",
                "description": "desc",
                "url": "https://newsapi.com/1",
                "source": {"name": "NewsAPI"},
                "publishedAt": "2026-03-12T10:00:00Z",
            }]
        }
        mock_get.return_value = mock_api_resp

        import feedparser
        rss_data = feedparser.parse("""<?xml version="1.0"?>
<rss><channel><item>
  <title>Markets rally on strong earnings</title>
  <description>same story</description>
  <link>https://rss.com/2</link>
</item></channel></rss>""")
        mock_parse.return_value = rss_data

        fetcher = NewsFetcher()
        fetcher.newsapi_key = "key"
        fetcher.gnews_key = ""
        fetcher.finnhub_key = ""
        rss = RSSFetcher(feeds={"Test": "http://fake/"})

        from scraper.news_fetcher import _deduplicate
        combined = _deduplicate(fetcher.fetch_all() + rss.fetch_all_rss())
        titles = [a.title for a in combined]
        self.assertEqual(titles.count("Markets rally on strong earnings"), 1)


if __name__ == "__main__":
    unittest.main()
