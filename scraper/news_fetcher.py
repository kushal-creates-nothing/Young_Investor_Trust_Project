"""
scraper/news_fetcher.py — Fetches financial news from NewsAPI, GNews, and Finnhub.

Gracefully degrades if API keys are missing or calls fail.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import requests

import config

logger = logging.getLogger(__name__)


@dataclass
class Article:
    title: str
    description: str
    url: str
    source: str
    published_at: str
    raw_text: str = ""


class NewsFetcher:
    """Fetches articles from NewsAPI, GNews, and Finnhub and deduplicates them."""

    def __init__(self):
        self.newsapi_key = config.NEWS_API_KEY
        self.gnews_key = config.GNEWS_API_KEY
        self.finnhub_key = config.FINNHUB_API_KEY

    # ── NewsAPI ────────────────────────────────────────────────────────────────

    def fetch_newsapi(self, query: str = "finance OR economy OR stock market", page_size: int = 20) -> List[Article]:
        """Call NewsAPI /v2/everything and return a list of Article objects."""
        if not self.newsapi_key:
            logger.warning("NEWS_API_KEY not set — skipping NewsAPI.")
            return []

        params = {
            "q": query,
            "pageSize": page_size,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": self.newsapi_key,
        }
        try:
            response = requests.get(config.NEWSAPI_ENDPOINT, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            articles = []
            for item in data.get("articles", []):
                articles.append(
                    Article(
                        title=item.get("title") or "",
                        description=item.get("description") or "",
                        url=item.get("url") or "",
                        source=item.get("source", {}).get("name") or "NewsAPI",
                        published_at=item.get("publishedAt") or "",
                        raw_text=(item.get("title") or "") + " " + (item.get("description") or ""),
                    )
                )
            logger.info("NewsAPI returned %d articles.", len(articles))
            return articles
        except requests.RequestException as exc:
            logger.error("NewsAPI request failed: %s", exc)
            return []

    # ── GNews ──────────────────────────────────────────────────────────────────

    def fetch_gnews(self, query: str = "finance stock market economy", max: int = 10) -> List[Article]:
        """Call GNews /api/v4/search and return a list of Article objects."""
        if not self.gnews_key:
            logger.warning("GNEWS_API_KEY not set — skipping GNews.")
            return []

        params = {
            "q": query,
            "max": max,
            "lang": "en",
            "token": self.gnews_key,
        }
        try:
            response = requests.get(config.GNEWS_ENDPOINT, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            articles = []
            for item in data.get("articles", []):
                articles.append(
                    Article(
                        title=item.get("title") or "",
                        description=item.get("description") or "",
                        url=item.get("url") or "",
                        source=item.get("source", {}).get("name") or "GNews",
                        published_at=item.get("publishedAt") or "",
                        raw_text=(item.get("title") or "") + " " + (item.get("description") or ""),
                    )
                )
            logger.info("GNews returned %d articles.", len(articles))
            return articles
        except requests.RequestException as exc:
            logger.error("GNews request failed: %s", exc)
            return []

    # ── Finnhub ────────────────────────────────────────────────────────────────

    def fetch_finnhub(self, category: str = "general") -> List[Article]:
        """Call Finnhub /api/v1/news and return a list of Article objects."""
        if not self.finnhub_key:
            logger.warning("FINNHUB_API_KEY not set — skipping Finnhub.")
            return []

        params = {
            "category": category,
            "token": self.finnhub_key,
        }
        try:
            response = requests.get(config.FINNHUB_ENDPOINT, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            articles = []
            for item in data:
                pub_dt = datetime.utcfromtimestamp(item.get("datetime", 0)).isoformat() if item.get("datetime") else ""
                articles.append(
                    Article(
                        title=item.get("headline") or "",
                        description=item.get("summary") or "",
                        url=item.get("url") or "",
                        source=item.get("source") or "Finnhub",
                        published_at=pub_dt,
                        raw_text=(item.get("headline") or "") + " " + (item.get("summary") or ""),
                    )
                )
            logger.info("Finnhub returned %d articles.", len(articles))
            return articles
        except requests.RequestException as exc:
            logger.error("Finnhub request failed: %s", exc)
            return []

    # ── Aggregate ──────────────────────────────────────────────────────────────

    def fetch_all(self) -> List[Article]:
        """Fetch from all three sources, deduplicate by URL and title, return unified list."""
        all_articles: List[Article] = []
        all_articles.extend(self.fetch_newsapi())
        all_articles.extend(self.fetch_gnews())
        all_articles.extend(self.fetch_finnhub())

        return _deduplicate(all_articles)


def _deduplicate(articles: List[Article]) -> List[Article]:
    """Remove duplicate articles by URL; fall back to normalised title comparison."""
    seen_urls: set = set()
    seen_titles: set = set()
    unique: List[Article] = []
    for article in articles:
        url_key = article.url.strip().lower()
        title_key = article.title.strip().lower()
        if url_key and url_key in seen_urls:
            continue
        if title_key and title_key in seen_titles:
            continue
        if url_key:
            seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        unique.append(article)
    return unique
