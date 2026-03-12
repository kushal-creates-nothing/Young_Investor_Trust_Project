"""
scraper/rss_fetcher.py — Fallback RSS scraper using feedparser.

Parses RSS feeds from Reuters, BBC Business, CNBC, and MarketWatch.
"""

import logging
from typing import List

import feedparser

import config
from scraper.news_fetcher import Article, _deduplicate

logger = logging.getLogger(__name__)


class RSSFetcher:
    """Fetches articles from configured RSS feeds."""

    def __init__(self, feeds: dict = None):
        self.feeds = feeds if feeds is not None else config.RSS_FEEDS

    def fetch_feed(self, name: str, url: str) -> List[Article]:
        """Parse a single RSS feed and return Article objects."""
        try:
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries:
                title = entry.get("title", "")
                description = entry.get("summary", "") or entry.get("description", "")
                link = entry.get("link", "")
                published = entry.get("published", "") or entry.get("updated", "")
                articles.append(
                    Article(
                        title=title,
                        description=description,
                        url=link,
                        source=name,
                        published_at=published,
                        raw_text=f"{title} {description}",
                    )
                )
            logger.info("RSS feed '%s' returned %d articles.", name, len(articles))
            return articles
        except Exception as exc:
            logger.error("Failed to fetch RSS feed '%s' (%s): %s", name, url, exc)
            return []

    def fetch_all_rss(self) -> List[Article]:
        """Fetch all configured RSS feeds and return a deduplicated list of Articles."""
        all_articles: List[Article] = []
        for name, url in self.feeds.items():
            all_articles.extend(self.fetch_feed(name, url))
        return _deduplicate(all_articles)
