import json
import logging
import os
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, create_engine,
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
    keywords_matched = Column(Text)
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

    def __init__(self, db_path=None):
        db_path = db_path or config.DB_PATH
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        logger.info("DB ready at '%s'", db_path)

    def save_articles(self, articles):
        url_to_id = {}
        with Session(self.engine) as session:
            for art in articles:
                if not art.url:
                    continue
                existing = session.query(ArticleModel).filter_by(url=art.url).first()
                if existing:
                    url_to_id[art.url] = existing.id
                else:
                    row = ArticleModel(
                        title=art.title,
                        description=art.description,
                        url=art.url,
                        source=art.source,
                        published_at=art.published_at,
                    )
                    session.add(row)
                    session.flush()
                    url_to_id[art.url] = row.id
            session.commit()
        return url_to_id

    def save_sentiment_results(self, results, url_to_id=None):
        with Session(self.engine) as session:
            for r in results:
                article_id = url_to_id.get(r.article.url) if url_to_id else None
                if article_id is None:
                    row = session.query(ArticleModel).filter_by(url=r.article.url).first()
                    if row:
                        article_id = row.id
                if article_id is None:
                    logger.warning("No DB row for '%s' — skipping", r.article.url)
                    continue
                session.add(SentimentResultModel(
                    article_id=article_id,
                    vader_compound=r.vader_compound,
                    vader_label=r.vader_label,
                    finbert_label=r.finbert_label,
                    finbert_score=r.finbert_score,
                    keywords_matched=json.dumps(r.keywords_matched),
                    is_safe_haven_signal=r.is_safe_haven_signal,
                    is_risk_on_signal=r.is_risk_on_signal,
                ))
            session.commit()

    def save_mood_snapshot(self, mood):
        with Session(self.engine) as session:
            session.add(MoodSnapshotModel(
                snapshot_time=datetime.utcnow(),
                overall_score=mood.get("overall_score"),
                shpi=mood.get("shpi"),
                risk_on_index=mood.get("risk_on_index"),
                fear_score=mood.get("fear_score"),
                mood_label=mood.get("mood_label"),
                tipping_point_alert=mood.get("tipping_point_alert", ""),
                article_count=mood.get("article_count", 0),
            ))
            session.commit()
        logger.info("Snapshot saved: %s", mood.get("mood_label"))

    def get_last_n_snapshots(self, n=24):
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
