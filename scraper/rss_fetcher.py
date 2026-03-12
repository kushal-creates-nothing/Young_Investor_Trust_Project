import logging
from typing import List

import feedparser

import config
from scraper.news_fetcher import Article, _deduplicate

logger = logging.getLogger(__name__)


class RSSFetcher:

    def __init__(self, feeds=None):
        self.feeds = feeds if feeds is not None else config.RSS_FEEDS

    def fetch_feed(self, name, url):
        try:
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries:
                title = entry.get("title", "")
                desc = entry.get("summary", "") or entry.get("description", "")
                link = entry.get("link", "")
                pub = entry.get("published", "") or entry.get("updated", "")
                articles.append(Article(
                    title=title,
                    description=desc,
                    url=link,
                    source=name,
                    published_at=pub,
                    raw_text=f"{title} {desc}",
                ))
            logger.info("RSS '%s': %d articles", name, len(articles))
            return articles
        except Exception as e:
            logger.error("RSS feed '%s' failed: %s", name, e)
            return []

    def fetch_all_rss(self):
        combined = []
        for name, url in self.feeds.items():
            combined.extend(self.fetch_feed(name, url))
        return _deduplicate(combined)
