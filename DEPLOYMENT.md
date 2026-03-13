# Deployment Guide

This project has two deployable components:

| Component | What it is | URL after deploy |
|---|---|---|
| **Static dashboard** | `docs/index.html` — pre-rendered demo with hardcoded data, no server needed | `https://<your-username>.github.io/Young_Investor_Trust_Project/` |
| **Live Flask app** | `app.py` — real-time analysis with live news APIs, AI chat, semantic search | whatever host you deploy it to, e.g. `https://your-app.onrender.com` |

---

## 1 — Deploy the static demo to GitHub Pages

The static dashboard in `docs/` is a single HTML file with all demo data baked in. It requires no server and no API keys. It is perfect for showing the project publicly.

### Step-by-step

1. **Push this branch / merge to `main`.**

2. Go to your repository on GitHub:  
   `https://github.com/<your-username>/Young_Investor_Trust_Project`

3. Click **Settings** → **Pages** (in the left sidebar under *Code and automation*).

4. Under **Build and deployment**:
   - **Source**: choose **Deploy from a branch**
   - **Branch**: `main`
   - **Folder**: `/docs`
   - Click **Save**

5. GitHub will build and deploy in ~1–2 minutes. You'll see a green banner with the live URL:  
   `https://<your-username>.github.io/Young_Investor_Trust_Project/`

6. Click **Visit site** (or the URL in the banner) to open the dashboard.

> **Note:** The `docs/.nojekyll` file (already committed) tells GitHub Pages to skip Jekyll processing and serve the file as-is. If you ever see a Jekyll build error, check that this file is present.

### What you'll see

The static demo shows a pre-populated Market Mood Dashboard:

- **Market Mood gauge** — sentiment score from −1 (extreme fear) to +1 (extreme optimism)
- **Safe-Haven Pressure bar** — how much gold/bond/recession coverage is in the news
- **Risk-On Activity bar** — bullish signals (IPOs, earnings beats, rallies)
- **Fear Score gauge**
- **Trend chart** — 13 historical snapshots showing mood over time
- **Headlines tabs** — Bearish / Bullish / Safe-Haven article lists
- **Notable Figures** — key people (Powell, Buffett, etc.) and their sentiment context
- **AI Chat** — a simulated RAG assistant (answers from hardcoded knowledge base in the demo)
- **Semantic Search** — search articles by topic (uses a static lookup table in the demo)
- **⚡ Run Analysis button** — simulates a pipeline run with a progress animation

> All data in the static demo is hardcoded for demonstration. To get live data you need the Flask app (see section 2).

---

## 2 — Run the live Flask app locally

The Flask app fetches real articles, runs NLP, stores results in SQLite, and serves an interactive dashboard with live data.

### Prerequisites

- Python 3.9+
- (Optional) API keys for NewsAPI, GNews, Finnhub — free tiers work fine

### Step-by-step

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Young_Investor_Trust_Project.git
cd Young_Investor_Trust_Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Open .env in a text editor and fill in at least one API key.
# Leave all keys blank to use RSS-only mode (no key required).

# 4. Seed demo data (skip if you have API keys and want real articles)
python seed_demo.py

# 5. Start the web server
python app.py
```

Open **http://localhost:5000** in your browser.

### Using the live dashboard

| Feature | How to use it |
|---|---|
| **⚡ Run Analysis** | Click the button on the dashboard to fetch fresh articles and re-score them immediately. Takes ~10–30 s depending on network. |
| **Trend chart** | Automatically updates after each run. Shows the last 24 mood snapshots. |
| **Headlines tabs** | Click Bearish / Bullish / Safe-Haven to filter articles. |
| **AI Chat** | Type any market question. Powered by RAG over the latest fetched articles. Requires `OPENAI_API_KEY` or `HUGGINGFACE_API_KEY` in `.env`. |
| **Semantic Search** | Type a topic (e.g. "recession", "tech rally") to find similar articles. |
| **Background scheduler** | Run `python scheduler.py` in a separate terminal to auto-fetch every 2 hours. |

### API endpoints (for developers)

| Endpoint | Description |
|---|---|
| `GET /api/latest` | Latest mood snapshot + headlines |
| `GET /api/history?n=24` | Last *n* mood snapshots |
| `GET /api/headlines` | Bearish / Bullish / Safe-Haven headline lists |
| `POST /api/run` | Trigger a pipeline run manually |
| `GET /api/status` | Pipeline status, vector store size |
| `POST /api/ask` | RAG-powered market question (`{"question": "..."}`) |
| `GET /api/search?q=<topic>&k=5` | Semantic article search |
| `GET /api/model-health` | MLOps health (drift, model registry, Triton) |

---

## 3 — Deploy the Flask app to a free cloud host

### Option A — Render (recommended, free tier available)

1. Go to [render.com](https://render.com) and sign in with GitHub.
2. Click **New → Web Service**.
3. Connect your `Young_Investor_Trust_Project` repository.
4. Fill in the settings:
   - **Name**: `young-investor-dashboard` (or anything you like)
   - **Runtime**: `Python 3`
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Under **Environment Variables**, add your API keys (same as `.env`).
6. Click **Create Web Service**.

Render will build and deploy automatically. Your live URL will be  
`https://young-investor-dashboard.onrender.com`.

> Install gunicorn first if it is not in `requirements.txt`: `pip install gunicorn` and add it to the file.

### Option B — Railway

1. Go to [railway.app](https://railway.app) and sign in with GitHub.
2. Click **New Project → Deploy from GitHub repo**.
3. Select `Young_Investor_Trust_Project`.
4. Railway auto-detects Python. Set the start command to:
   ```
   gunicorn app:app --bind 0.0.0.0:$PORT
   ```
5. Add environment variables in the **Variables** tab.
6. Click **Deploy**. Your URL appears in the dashboard.

### Option C — Heroku

```bash
# Install Heroku CLI, then:
heroku create young-investor-dashboard
heroku config:set NEWS_API_KEY=xxx GNEWS_API_KEY=xxx FINNHUB_API_KEY=xxx
git push heroku main
heroku open
```

Make sure a `Procfile` exists in the repo root:
```
web: gunicorn app:app
```

---

## 4 — Difference between the static demo and the live app

| | Static GitHub Pages | Live Flask app |
|---|---|---|
| **Data** | Hardcoded demo data | Real-time news from APIs |
| **Run Analysis button** | Visual simulation only | Actually fetches articles |
| **AI Chat** | Simulated responses | Real RAG over fetched articles |
| **Semantic Search** | Keyword lookup table | Vector similarity search (ChromaDB) |
| **Requires server** | No | Yes |
| **API keys needed** | No | Optional (RSS fallback works without) |
| **Best for** | Sharing / portfolio | Actual use |

---

## 5 — Tipping Point alert

The dashboard fires an orange **⚡ Tipping Point** banner when:

```
Safe-Haven Pressure Index > 40%  AND  Overall Sentiment < −0.2
```

This signals that news coverage is pushing retail investors toward "safe" assets (gold, bonds) instead of stocks. It is not financial advice — it is a sentiment signal.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| GitHub Pages shows Jekyll error | Make sure `docs/.nojekyll` exists (it is committed in this repo) |
| `http://localhost:5000` shows "No data yet" | Run `python seed_demo.py` or click **⚡ Run Analysis** |
| Run Analysis takes a long time | Normal — it fetches from live APIs. Click once and wait ~30 s |
| AI Chat says "run pipeline first" | Click **⚡ Run Analysis** to populate the vector store, then ask again |
| RSS only, no API articles | Add at least one key in `.env` — or leave as-is, RSS fallback works fine |
