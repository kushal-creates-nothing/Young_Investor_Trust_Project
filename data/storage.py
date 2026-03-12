"""
data/storage.py — SQLite persistence layer using SQLAlchemy.

Tables:
  - articles         : raw article metadata
  - sentiment_results: per-article sentiment scores
  - mood_snapshots   : aggregated market mood per run
"""

import json
import logging
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

import config

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class ArticleModel(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(Text, nullable=False)
    description = Column(Text)
    url = Column(String(2048), unique=True, nullable=False, index=True)
    source = Column(String(256))
    published_at = Column(String(64))
    fetched_at = Column(DateTime, default=datetime.utcnow)

    sentiment_results = relationship("SentimentResultModel", back_populates="article", cascade="all, delete-orphan")


class SentimentResultModel(Base):
    __tablename__ = "sentiment_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=False, index=True)
    vader_compound = Column(Float)
    vader_label = Column(String(16))
    finbert_label = Column(String(16))
    finbert_score = Column(Float)
    keywords_matched = Column(Text)   # JSON-encoded list
    is_safe_haven_signal = Column(Boolean, default=False)
    is_risk_on_signal = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    article = relationship("ArticleModel", back_populates="sentiment_results")


class MoodSnapshotModel(Base):
    __tablename__ = "mood_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_time = Column(DateTime, default=datetime.utcnow, index=True)
    overall_score = Column(Float)
    shpi = Column(Float)
    risk_on_index = Column(Float)
    fear_score = Column(Float)
    mood_label = Column(String(64))
    tipping_point_alert = Column(Text)
    article_count = Column(Integer)


class DataStorage:
    """Manages all database read/write operations."""

    def __init__(self, db_path: str = None):
        db_path = db_path or config.DB_PATH
        # Ensure the parent directory exists
        import os
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        logger.info("DataStorage initialised at '%s'.", db_path)

    # ── Articles ───────────────────────────────────────────────────────────────

    def save_articles(self, articles) -> dict:
        """Upsert articles by URL. Returns mapping url -> article_id."""
        url_to_id: dict = {}
        with Session(self.engine) as session:
            for article in articles:
                if not article.url:
                    continue
                existing = session.query(ArticleModel).filter_by(url=article.url).first()
                if existing:
                    url_to_id[article.url] = existing.id
                else:
                    new_article = ArticleModel(
                        title=article.title,
                        description=article.description,
                        url=article.url,
                        source=article.source,
                        published_at=article.published_at,
                    )
                    session.add(new_article)
                    session.flush()
                    url_to_id[article.url] = new_article.id
            session.commit()
        return url_to_id

    # ── Sentiment Results ──────────────────────────────────────────────────────

    def save_sentiment_results(self, results, url_to_id: dict = None):
        """Persist sentiment results. url_to_id maps article URL to DB article id."""
        with Session(self.engine) as session:
            for result in results:
                article_id = None
                if url_to_id:
                    article_id = url_to_id.get(result.article.url)
                if article_id is None:
                    # Fallback: look up by URL
                    row = session.query(ArticleModel).filter_by(url=result.article.url).first()
                    if row:
                        article_id = row.id
                if article_id is None:
                    logger.warning("No DB article found for URL '%s' — skipping sentiment.", result.article.url)
                    continue

                sr = SentimentResultModel(
                    article_id=article_id,
                    vader_compound=result.vader_compound,
                    vader_label=result.vader_label,
                    finbert_label=result.finbert_label,
                    finbert_score=result.finbert_score,
                    keywords_matched=json.dumps(result.keywords_matched),
                    is_safe_haven_signal=result.is_safe_haven_signal,
                    is_risk_on_signal=result.is_risk_on_signal,
                )
                session.add(sr)
            session.commit()

    # ── Mood Snapshots ─────────────────────────────────────────────────────────

    def save_mood_snapshot(self, mood_dict: dict):
        """Persist a mood snapshot."""
        with Session(self.engine) as session:
            snapshot = MoodSnapshotModel(
                snapshot_time=datetime.utcnow(),
                overall_score=mood_dict.get("overall_score"),
                shpi=mood_dict.get("shpi"),
                risk_on_index=mood_dict.get("risk_on_index"),
                fear_score=mood_dict.get("fear_score"),
                mood_label=mood_dict.get("mood_label"),
                tipping_point_alert=mood_dict.get("tipping_point_alert", ""),
                article_count=mood_dict.get("article_count", 0),
            )
            session.add(snapshot)
            session.commit()
        logger.info("Mood snapshot saved: %s", mood_dict.get("mood_label"))

    def get_last_n_snapshots(self, n: int = 24) -> list:
        """Return the last n mood snapshots as dicts, ordered newest-first."""
        with Session(self.engine) as session:
            rows = (
                session.query(MoodSnapshotModel)
                .order_by(MoodSnapshotModel.snapshot_time.desc())
                .limit(n)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "snapshot_time": r.snapshot_time.isoformat() if r.snapshot_time else "",
                    "overall_score": r.overall_score,
                    "shpi": r.shpi,
                    "risk_on_index": r.risk_on_index,
                    "fear_score": r.fear_score,
                    "mood_label": r.mood_label,
                    "tipping_point_alert": r.tipping_point_alert,
                    "article_count": r.article_count,
                }
                for r in rows
            ]
