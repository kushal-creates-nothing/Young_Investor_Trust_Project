import unittest
from unittest.mock import MagicMock, patch

from scraper.news_fetcher import Article
from sentiment.analyzer import SentimentResult


def make_article(title, url="https://test.com/a1", source="Test"):
    return Article(title=title, description="", url=url, source=source,
                   published_at="2026-03-12T00:00:00Z", raw_text=title)


def make_result(title, compound, is_safe=False, url_suffix="1"):
    label = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")
    art = make_article(title, url=f"https://test.com/{url_suffix}")
    return SentimentResult(art, compound, label, label, abs(compound), [], is_safe, False)


class TestFinancialNewsEmbedder(unittest.TestCase):

    def setUp(self):
        from rag.embedder import FinancialNewsEmbedder
        self.embedder = FinancialNewsEmbedder.__new__(FinancialNewsEmbedder)
        # force TF-IDF backend so we don't need torch in CI
        self.embedder._backend = "tfidf"
        self.embedder._model = None
        self.embedder._fitted = False

    def test_embed_documents_returns_list_of_lists(self):
        texts = ["Gold prices surge", "Markets rally on earnings", "Recession fears grow"]
        vecs = self.embedder.embed_documents(texts)
        self.assertEqual(len(vecs), 3)
        self.assertIsInstance(vecs[0], list)
        self.assertGreater(len(vecs[0]), 0)

    def test_embed_query_returns_list(self):
        # need to fit first
        self.embedder.embed_documents(["Gold prices surge", "Markets rally"])
        vec = self.embedder.embed_query("gold safe haven")
        self.assertIsInstance(vec, list)
        self.assertGreater(len(vec), 0)

    def test_embeddings_are_normalised(self):
        import math
        texts = ["The market fell sharply on recession news", "Bull market confirmed"]
        vecs = self.embedder.embed_documents(texts)
        for v in vecs:
            norm = math.sqrt(sum(x * x for x in v))
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_same_text_produces_same_embedding(self):
        texts = ["Gold hits record high"]
        v1 = self.embedder.embed_documents(texts)
        v2 = self.embedder.embed_documents(texts)
        self.assertEqual(v1, v2)

    def test_different_texts_produce_different_embeddings(self):
        self.embedder.embed_documents(["Gold prices surge", "Markets crash on fear",
                                        "IPO season heats up", "Fed raises rates"])
        v1 = self.embedder.embed_query("gold safe haven")
        v2 = self.embedder.embed_query("tech stocks rally earnings beat")
        self.assertNotEqual(v1, v2)

    def test_empty_list_returns_empty(self):
        result = self.embedder.embed_documents(["seed doc for vocab"])
        empty = self.embedder.embed_documents([])
        self.assertEqual(empty, [])


class TestArticleVectorStore(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()

    @patch("langchain_community.vectorstores.Chroma")
    def test_add_articles_calls_store(self, mock_chroma_cls):
        mock_store = MagicMock()
        mock_chroma_cls.return_value = mock_store

        from rag.vector_store import ArticleVectorStore
        from rag.embedder import FinancialNewsEmbedder

        embedder = MagicMock(spec=FinancialNewsEmbedder)
        vs = ArticleVectorStore(persist_dir=self.tmp, embedder=embedder)

        articles = [make_article("Gold rises", url="https://test.com/v1")]
        results = [make_result("Gold rises", -0.3, is_safe=True, url_suffix="v1")]
        vs.add_articles(articles, results)

        mock_store.add_texts.assert_called_once()

    @patch("langchain_community.vectorstores.Chroma")
    def test_add_empty_articles_returns_zero(self, mock_chroma_cls):
        from rag.vector_store import ArticleVectorStore
        vs = ArticleVectorStore(persist_dir=self.tmp, embedder=MagicMock())
        count = vs.add_articles([], [])
        self.assertEqual(count, 0)

    @patch("langchain_community.vectorstores.Chroma")
    def test_similarity_search_returns_list(self, mock_chroma_cls):
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = [
            (MagicMock(metadata={"title": "Gold hits high"}, page_content="Gold"), 0.9),
        ]
        mock_chroma_cls.return_value = mock_store

        from rag.vector_store import ArticleVectorStore
        vs = ArticleVectorStore(persist_dir=self.tmp, embedder=MagicMock())
        results = vs.similarity_search("gold safe haven", k=3)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0][1], 0.9)

    @patch("langchain_community.vectorstores.Chroma")
    def test_similarity_search_handles_error(self, mock_chroma_cls):
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.side_effect = Exception("DB error")
        mock_chroma_cls.return_value = mock_store

        from rag.vector_store import ArticleVectorStore
        vs = ArticleVectorStore(persist_dir=self.tmp, embedder=MagicMock())
        results = vs.similarity_search("gold")
        self.assertEqual(results, [])

    @patch("langchain_community.vectorstores.Chroma")
    def test_as_retriever_returns_retriever(self, mock_chroma_cls):
        mock_store = MagicMock()
        mock_chroma_cls.return_value = mock_store

        from rag.vector_store import ArticleVectorStore
        vs = ArticleVectorStore(persist_dir=self.tmp, embedder=MagicMock())
        vs.as_retriever(k=3)
        mock_store.as_retriever.assert_called_once()


class TestMarketInsightRAG(unittest.TestCase):

    def _make_doc(self, title, label, compound, is_safe="False"):
        doc = MagicMock()
        doc.metadata = {"title": title, "source": "Test", "url": "https://t.com/1",
                        "vader_label": label, "vader_compound": compound, "is_safe_haven": is_safe}
        doc.page_content = title
        return doc

    def test_template_llm_returns_string(self):
        from rag.rag_chain import _TemplateLLM
        llm = _TemplateLLM()
        result = llm._analyse("Source: Reuters] Gold prices surge\n  Sentiment: negative (-0.7) | Safe-haven: True\n")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 20)

    def test_template_llm_detects_safe_haven(self):
        from rag.rag_chain import _TemplateLLM
        llm = _TemplateLLM()
        context = "\n".join([
            "[Reuters] Gold surge\n  Sentiment: negative (-0.7) | Safe-haven: True",
            "[BBC] Bonds rally\n  Sentiment: negative (-0.5) | Safe-haven: True",
            "[CNBC] Cash demand rises\n  Sentiment: negative (-0.4) | Safe-haven: True",
        ])
        result = llm._analyse(context)
        # should mention safety/caution
        self.assertTrue(any(word in result.lower() for word in ["safe", "caution", "gold", "bond"]))

    def test_rag_chain_ask_with_mock_retriever(self):
        from rag.rag_chain import MarketInsightRAG, _TemplateLLM

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            self._make_doc("Gold prices hit record", "negative", -0.7, "True"),
            self._make_doc("Markets rally on earnings", "positive", 0.6, "False"),
        ]

        chain = MarketInsightRAG(retriever=mock_retriever, llm=_TemplateLLM())
        result = chain.ask("Should I buy gold now?")

        self.assertIn("question", result)
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIsInstance(result["answer"], str)
        self.assertGreater(len(result["answer"]), 10)

    def test_rag_chain_handles_retriever_failure(self):
        from rag.rag_chain import MarketInsightRAG, _TemplateLLM

        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = Exception("vector DB down")

        chain = MarketInsightRAG(retriever=mock_retriever, llm=_TemplateLLM())
        result = chain.ask("What is happening in markets?")
        self.assertIn("answer", result)
        # should return the error fallback, not crash
        self.assertIsInstance(result["answer"], str)

    def test_format_docs(self):
        from rag.rag_chain import _format_docs
        docs = [
            self._make_doc("Gold up 3%", "negative", -0.6, "True"),
            self._make_doc("Tech rally", "positive", 0.7, "False"),
        ]
        formatted = _format_docs(docs)
        self.assertIn("Gold up 3%", formatted)
        self.assertIn("Tech rally", formatted)
        self.assertIn("-0.6", formatted)


if __name__ == "__main__":
    unittest.main()
