import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

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

    def __init__(self):
        self.newsapi_key = config.NEWS_API_KEY
        self.gnews_key = config.GNEWS_API_KEY
        self.finnhub_key = config.FINNHUB_API_KEY

    def fetch_newsapi(self, query="finance OR economy OR stock market", page_size=20):
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
            resp = requests.get(config.NEWSAPI_ENDPOINT, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            out = []
            for item in data.get("articles", []):
                title = item.get("title") or ""
                desc = item.get("description") or ""
                out.append(Article(
                    title=title,
                    description=desc,
                    url=item.get("url") or "",
                    source=item.get("source", {}).get("name") or "NewsAPI",
                    published_at=item.get("publishedAt") or "",
                    raw_text=f"{title} {desc}",
                ))
            logger.info("NewsAPI: got %d articles", len(out))
            return out
        except requests.RequestException as e:
            logger.error("NewsAPI failed: %s", e)
            return []

    def fetch_gnews(self, query="finance stock market economy", max=10):
        if not self.gnews_key:
            logger.warning("GNEWS_API_KEY not set — skipping GNews.")
            return []
        params = {"q": query, "max": max, "lang": "en", "token": self.gnews_key}
        try:
            resp = requests.get(config.GNEWS_ENDPOINT, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            out = []
            for item in data.get("articles", []):
                title = item.get("title") or ""
                desc = item.get("description") or ""
                out.append(Article(
                    title=title,
                    description=desc,
                    url=item.get("url") or "",
                    source=item.get("source", {}).get("name") or "GNews",
                    published_at=item.get("publishedAt") or "",
                    raw_text=f"{title} {desc}",
                ))
            logger.info("GNews: got %d articles", len(out))
            return out
        except requests.RequestException as e:
            logger.error("GNews failed: %s", e)
            return []

    def fetch_finnhub(self, category="general"):
        if not self.finnhub_key:
            logger.warning("FINNHUB_API_KEY not set — skipping Finnhub.")
            return []
        params = {"category": category, "token": self.finnhub_key}
        try:
            resp = requests.get(config.FINNHUB_ENDPOINT, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            out = []
            for item in data:
                ts = item.get("datetime", 0)
                pub = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else ""
                headline = item.get("headline") or ""
                summary = item.get("summary") or ""
                out.append(Article(
                    title=headline,
                    description=summary,
                    url=item.get("url") or "",
                    source=item.get("source") or "Finnhub",
                    published_at=pub,
                    raw_text=f"{headline} {summary}",
                ))
            logger.info("Finnhub: got %d articles", len(out))
            return out
        except requests.RequestException as e:
            logger.error("Finnhub failed: %s", e)
            return []

    def fetch_all(self):
        combined = []
        combined.extend(self.fetch_newsapi())
        combined.extend(self.fetch_gnews())
        combined.extend(self.fetch_finnhub())
        return _deduplicate(combined)


def _deduplicate(articles: List[Article]) -> List[Article]:
    seen_urls = set()
    seen_titles = set()
    result = []
    for a in articles:
        url_key = a.url.strip().lower()
        title_key = a.title.strip().lower()
        if url_key and url_key in seen_urls:
            continue
        if title_key and title_key in seen_titles:
            continue
        if url_key:
            seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        result.append(a)
    return result
