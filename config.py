"""
config.py — Central configuration loader for the Young Investor Trust Project.

Loads environment variables via python-dotenv and defines all constants
used across the project (API endpoints, keyword lists, safe-haven assets, etc.).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ────────────────────────────────────────────────────────────────
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
GNEWS_API_KEY: str = os.getenv("GNEWS_API_KEY", "")
FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")

# ─── Behaviour ────────────────────────────────────────────────────────────────
FETCH_INTERVAL_HOURS: int = int(os.getenv("FETCH_INTERVAL_HOURS", "2"))
DB_PATH: str = os.getenv("DB_PATH", "data/sentiment_data.db")
USE_FINBERT: bool = os.getenv("USE_FINBERT", "false").lower() == "true"

# ─── API Endpoints ────────────────────────────────────────────────────────────
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"
FINNHUB_ENDPOINT = "https://finnhub.io/api/v1/news"

# ─── RSS Feed URLs ─────────────────────────────────────────────────────────────
RSS_FEEDS = {
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "BBC Business": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "CNBC Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch": "http://feeds.marketwatch.com/marketwatch/topstories/",
}

# ─── Sentiment Keywords ───────────────────────────────────────────────────────
RISK_OFF_KEYWORDS = [
    "gold",
    "safe haven",
    "recession",
    "inflation",
    "market crash",
    "sell-off",
    "fear",
    "uncertainty",
    "central bank",
    "interest rate hike",
    "bear market",
]

RISK_ON_KEYWORDS = [
    "bull market",
    "growth",
    "rally",
    "earnings beat",
    "ipo",
    "optimism",
    "investment",
    "gdp growth",
]

NOTABLE_FIGURES = [
    "Warren Buffett",
    "Jerome Powell",
    "Elon Musk",
    "Janet Yellen",
    "IMF",
    "World Bank",
    "Federal Reserve",
]

# ─── Safe-Haven Assets ────────────────────────────────────────────────────────
SAFE_HAVEN_ASSETS = ["gold", "bonds", "treasury", "cash", "yen", "swiss franc", "silver"]

# ─── Tipping-Point Thresholds ─────────────────────────────────────────────────
SHPI_TIPPING_THRESHOLD = 0.4   # Safe-Haven Pressure Index threshold
SENTIMENT_TIPPING_THRESHOLD = -0.2  # Overall sentiment threshold
