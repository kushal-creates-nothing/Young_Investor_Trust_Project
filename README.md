# Young Investor Trust — Market Mood Dashboard

A financial news sentiment analysis tool that scrapes articles from multiple sources every 2 hours, scores them with NLP, and shows young investors where the market mood is heading — with a particular focus on detecting when sentiment is pushing people toward "safe" assets like gold or bonds instead of stocks.

---

## What it does

The tool pulls news from NewsAPI, GNews, Finnhub, Reuters, BBC Business, CNBC, and MarketWatch. It runs each headline through VADER sentiment scoring (and optionally FinBERT for deeper financial NLP). The results feed a set of metrics:

- **Overall Sentiment Score** — the average mood of all articles, from -1 (extreme fear) to +1 (extreme optimism)
- **Safe-Haven Pressure Index (SHPI)** — what fraction of articles mention gold, bonds, recession, etc.
- **Risk-On Index** — what fraction mention bullish signals like IPOs, rallies, earnings beats
- **Fear Score** — a blended fear/greed indicator
- **Tipping Point Alert** — fires when SHPI > 40% AND overall sentiment < -0.2, meaning the crowd may be quietly fleeing equities

The interactive web dashboard makes all of this visual and accessible to people with no finance background.

---

## Project structure

```
Young_Investor_Trust_Project/
├── app.py                   ← Flask web server (the website)
├── scheduler.py             ← Background pipeline runner (every 2 hours)
├── seed_demo.py             ← Loads demo data so the site works without API keys
├── config.py                ← Env vars + constants
├── requirements.txt
├── .env.example
├── scraper/
│   ├── news_fetcher.py      ← NewsAPI, GNews, Finnhub
│   └── rss_fetcher.py       ← RSS fallback (Reuters, BBC, CNBC, MarketWatch)
├── sentiment/
│   ├── analyzer.py          ← VADER + optional FinBERT
│   └── scoring.py           ← Mood score, SHPI, tipping point logic
├── data/
│   └── storage.py           ← SQLite via SQLAlchemy
├── dashboard/
│   └── report.py            ← JSON, text, and HTML reports
├── templates/
│   └── index.html           ← Interactive web dashboard
├── tests/
│   ├── test_scraper.py
│   ├── test_sentiment.py
│   └── test_scoring.py
└── reports/                 ← Auto-generated JSON + HTML snapshots
```

---

## Deployment

For step-by-step instructions on how to deploy both the **static GitHub Pages demo** and the **live Flask app** to the cloud, see **[DEPLOYMENT.md](DEPLOYMENT.md)**.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys (optional)

Copy `.env.example` to `.env` and fill in your keys. All three APIs have free tiers.

| Key | Where to get it |
|---|---|
| `NEWS_API_KEY` | https://newsapi.org (free tier: 100 req/day) |
| `GNEWS_API_KEY` | https://gnews.io (free tier: 100 req/day) |
| `FINNHUB_API_KEY` | https://finnhub.io (free tier: 60 req/min) |

You can leave all keys blank — the RSS fallback feeds will still work without any keys.

### 3. Load demo data (skip if you have API keys)

```bash
python seed_demo.py
```

This seeds the database with 20 realistic demo articles and 13 historical mood snapshots so the dashboard has something to show immediately.

### 4. Start the website

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 5. (Optional) Run the background scheduler

In a separate terminal:

```bash
python scheduler.py
```

This fetches fresh articles and updates the database every 2 hours automatically. The website will show the updated data on the next page refresh or when you click **Run Now**.

---

## Using the dashboard

The dashboard is designed for anyone — no finance knowledge needed.

- **Market Mood card** — the big number and label tells you the overall vibe of today's news
- **Safe-Haven Pressure bar** — rising bar means more people are talking about gold and bonds
- **Risk-On Activity bar** — rising bar means more bullish news (IPOs, earnings, growth)
- **Fear Score** — straightforward 0–100% fear gauge
- **⚡ Run Now** — triggers a fresh analysis immediately without waiting 2 hours
- **Headlines tabs** — flip between Bearish, Bullish, and Safe-Haven articles
- **Trend chart** — see how mood has shifted across the last 24 runs
- **People in the News** — shows whether key figures like Jerome Powell or Warren Buffett are being discussed positively or negatively

---

## Understanding the Tipping Point

The orange alert banner fires when:

```
Safe-Haven Pressure Index > 40%  AND  Overall Sentiment < -0.2
```

This combination is the signal that collective news sentiment is pushing retail investors toward safety. It doesn't mean you should act on it — but it's worth paying attention to.

The mood labels map roughly like this:

| Label | What it means |
|---|---|
| Strongly Bullish | Very positive news — investors are excited |
| Bullish | More good news than bad |
| Neutral | Mixed bag — no strong signal either way |
| Bearish | More bad news than good |
| Strongly Bearish | Lots of fear and negativity in the headlines |
| Safe-Haven Flight | High fear + high gold/bond mentions — crowd may be heading for the exit |

---

## Running FinBERT (optional, slow)

FinBERT is a financial-domain BERT model that gives deeper sentiment scoring. It's disabled by default because it requires ~500MB download and a few seconds per article.

To enable it:
```bash
# in your .env
USE_FINBERT=true
```

First run will download the model automatically. Subsequent runs use the cached version.

---

## Running tests

```bash
python -m pytest tests/ -v
# or without pytest:
python -m unittest discover tests/
```

---

## Sample output (terminal)

```
╔══════════════════════════════════════════════════════════╗
║       🧠 YOUNG INVESTOR MARKET MOOD REPORT               ║
║       📅 2026-03-12 14:00 UTC                            ║
╠══════════════════════════════════════════════════════════╣
║  Overall Sentiment:   🔴 Bearish              (-0.31)    ║
║  Safe-Haven Pressure: 🟡 Moderate             (0.38)     ║
║  Risk-On Activity:    🔴 Low                  (0.14)     ║
║  Fear Score:          0.62 / 1.00                        ║
╠══════════════════════════════════════════════════════════╣
║  ⚠️ TIPPING POINT APPROACHING — monitor closely          ║
╚══════════════════════════════════════════════════════════╝

📰 TOP NEGATIVE HEADLINES:
  1. "Fed signals further rate hikes amid persistent inflation" (-0.72)
  2. "Global markets tumble on recession fears" (-0.68)

📰 TOP POSITIVE HEADLINES:
  1. "Tech rally lifts markets as earnings beat expectations" (+0.65)

🛡️  Safe-haven signals: 9 / 20 articles

👤 NOTABLE FIGURES MENTIONED:
  Jerome Powell — negative context (3 articles)
  Warren Buffett — neutral context (1 article)
```
