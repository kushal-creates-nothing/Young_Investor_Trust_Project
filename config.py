import os
from dotenv import load_dotenv

load_dotenv()

# API keys — grab from .env, empty string means that source gets skipped
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

FETCH_INTERVAL_HOURS = int(os.getenv("FETCH_INTERVAL_HOURS", "2"))
DB_PATH = os.getenv("DB_PATH", "data/sentiment_data.db")
USE_FINBERT = os.getenv("USE_FINBERT", "false").lower() == "true"

# API base URLs
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"
FINNHUB_ENDPOINT = "https://finnhub.io/api/v1/news"

RSS_FEEDS = {
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "BBC Business": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "CNBC Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch": "http://feeds.marketwatch.com/marketwatch/topstories/",
}

# words/phrases that suggest people are getting nervous and fleeing to safety
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

# words that suggest confidence and risk appetite
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

SAFE_HAVEN_ASSETS = ["gold", "bonds", "treasury", "cash", "yen", "swiss franc", "silver"]

# tipping point thresholds — tuned by observation, adjust if needed
SHPI_TIPPING_THRESHOLD = 0.4
SENTIMENT_TIPPING_THRESHOLD = -0.2

# ── research-grade stack ───────────────────────────────────────────────────────

# RAG / Vector DB
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# MLOps
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "young-investor-trust")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")
MODEL_REGISTRY_PATH = os.getenv("MODEL_REGISTRY_PATH", "data/model_registry.json")

# Triton Inference Server
TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "finbert")
USE_TRITON = os.getenv("USE_TRITON", "false").lower() == "true"

# Fine-tuned model path (used by analyzer when available)
FINETUNED_MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH", "models/finbert-finetuned")
