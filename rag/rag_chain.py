import logging
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import config

logger = logging.getLogger(__name__)

# ── prompt ─────────────────────────────────────────────────────────────────────
# Designed to produce plain-English answers that young investors actually
# understand — no jargon, no walls of text.
_PROMPT = ChatPromptTemplate.from_template(
    """You are a financial analyst assistant helping young investors understand market conditions.
Keep your answer simple, clear and jargon-free.

Recent news articles most relevant to the question:
{context}

Question: {question}

Respond with:
1. A direct plain-English answer (2-3 sentences)
2. The key market signal this question highlights
3. What a cautious young investor should watch for
"""
)


def _format_docs(docs) -> str:
    parts = []
    for doc in docs:
        m = doc.metadata
        score = m.get("vader_compound", "n/a")
        label = m.get("vader_label", "unknown")
        parts.append(
            f"[{m.get('source','?')}] {m.get('title', doc.page_content[:100])}\n"
            f"  Sentiment: {label} ({score}) | Safe-haven: {m.get('is_safe_haven','false')}"
        )
    return "\n\n".join(parts) if parts else "No relevant articles found."


# ── LLM selection ──────────────────────────────────────────────────────────────

def _build_llm():
    """Pick the best available LLM. Gracefully degrades through the options."""

    # 1. OpenAI GPT — best quality
    if getattr(config, "OPENAI_API_KEY", ""):
        try:
            from langchain_openai import ChatOpenAI
            logger.info("RAG LLM: OpenAI gpt-3.5-turbo")
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                openai_api_key=config.OPENAI_API_KEY,
            )
        except Exception as e:
            logger.warning("OpenAI LLM init failed: %s", e)

    # 2. HuggingFace Inference API — free tier
    if getattr(config, "HUGGINGFACE_API_KEY", ""):
        try:
            from langchain_huggingface import HuggingFaceEndpoint
            logger.info("RAG LLM: HuggingFace flan-t5-large")
            return HuggingFaceEndpoint(
                repo_id="google/flan-t5-large",
                task="text2text-generation",
                huggingfacehub_api_token=config.HUGGINGFACE_API_KEY,
                max_new_tokens=300,
            )
        except Exception as e:
            logger.warning("HuggingFace LLM init failed: %s", e)

    # 3. Template fallback — no API key needed
    logger.info("RAG LLM: template-based fallback (no API key configured)")
    return _TemplateLLM()


# ── template fallback ──────────────────────────────────────────────────────────
# This "LLM" generates structured insights from article metadata alone,
# so the RAG pipeline is useful even without any API key.

class _TemplateLLM:
    """Fake LLM that generates structured market insights from the context string."""

    def invoke(self, prompt_value) -> str:
        text = prompt_value.text if hasattr(prompt_value, "text") else str(prompt_value)
        return self._analyse(text)

    def __or__(self, other):
        # support LCEL pipe operator: llm | StrOutputParser()
        class _Piped:
            def __init__(self, llm, parser):
                self._llm = llm
                self._parser = parser

            def invoke(self, x):
                result = self._llm.invoke(x)
                return self._parser.invoke(result) if hasattr(self._parser, "invoke") else result

        return _Piped(self, other)

    def _analyse(self, context: str) -> str:
        lines = context.split("\n")
        safe_count = sum(1 for l in lines if "Safe-haven: True" in l)
        neg_count = sum(1 for l in lines if "negative" in l.lower())
        pos_count = sum(1 for l in lines if "positive" in l.lower())
        total = max(safe_count + neg_count + pos_count, 1)

        if safe_count / total > 0.5:
            mood = "cautious and risk-averse"
            advice = "Gold and bonds are gaining attention — consider reviewing how much risk you're comfortable with."
        elif pos_count / total > 0.5:
            mood = "optimistic and growth-focused"
            advice = "Market sentiment looks positive. Good time to research quality stocks, but always diversify."
        else:
            mood = "mixed and uncertain"
            advice = "The market is sending mixed signals. Staying diversified and watching key data releases is wise."

        safe_str = f"{safe_count} out of the top {len([l for l in lines if l.strip().startswith('[')])} retrieved articles mention safe-haven assets"
        return (
            f"Based on recent news most relevant to your question: market sentiment is currently {mood}. "
            f"{safe_str}. {advice}\n\n"
            f"Key signal: {'Safe-haven flight risk is elevated' if safe_count/total > 0.4 else 'No extreme fear signals detected — market is functioning normally'}.\n\n"
            f"Watch for: Fed rate decisions, CPI data releases, and gold price movements as leading indicators."
        )


# ── main class ─────────────────────────────────────────────────────────────────

class MarketInsightRAG:
    """
    LangChain LCEL-based RAG chain for answering market questions.

    Architecture:
        User question
            ↓
        ChromaDB semantic retrieval (top-5 articles)
            ↓
        Context formatting
            ↓
        LLM (OpenAI → HuggingFace → template fallback)
            ↓
        Plain-English answer + source articles
    """

    def __init__(self, retriever, llm=None):
        self._retriever = retriever
        self._llm = llm or _build_llm()
        self._chain = self._build_chain()

    def _build_chain(self):
        fmt = RunnableLambda(_format_docs)
        context = self._retriever | fmt

        if isinstance(self._llm, _TemplateLLM):
            # template LLM doesn't use the ChatPromptTemplate properly,
            # so build a simpler chain manually
            def _invoke(question):
                docs = self._retriever.invoke(question)
                ctx = _format_docs(docs)
                full = f"Context:\n{ctx}\n\nQuestion: {question}"
                return self._llm._analyse(full)
            return _invoke

        chain = (
            {"context": context, "question": RunnablePassthrough()}
            | _PROMPT
            | self._llm
            | StrOutputParser()
        )
        return chain.invoke

    def ask(self, question: str) -> dict:
        """Run RAG chain. Returns answer + retrieved source articles."""
        try:
            answer = self._chain(question)
            # fetch sources separately so we can return them with metadata
            sources = []
            try:
                raw_docs = self._retriever.invoke(question)
                for doc in raw_docs:
                    sources.append({
                        "title": doc.metadata.get("title", ""),
                        "source": doc.metadata.get("source", ""),
                        "url": doc.metadata.get("url", ""),
                        "sentiment": doc.metadata.get("vader_label", ""),
                        "score": doc.metadata.get("vader_compound", 0),
                    })
            except Exception:
                pass
            return {"question": question, "answer": answer, "sources": sources}
        except Exception as e:
            logger.error("RAG chain failed: %s", e)
            return {
                "question": question,
                "answer": "Analysis temporarily unavailable — please try again shortly.",
                "sources": [],
            }
