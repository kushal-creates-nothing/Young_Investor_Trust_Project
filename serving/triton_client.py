import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# try to import tritonclient — it's optional, we gracefully fall back
try:
    import tritonclient.http as httpclient
    _HAVE_TRITON = True
except ImportError:
    _HAVE_TRITON = False
    logger.info("tritonclient not installed — Triton serving unavailable, using local inference")


class TritonSentimentClient:
    """
    Client for the NVIDIA Triton Inference Server running the fine-tuned FinBERT model.

    When Triton is available and the server is running, this client sends tokenized
    text as gRPC/HTTP requests and gets back sentiment logits.

    When Triton is NOT available (not installed, server not running), it falls back
    to local HuggingFace inference so the rest of the pipeline keeps working.

    To deploy FinBERT on Triton:
      1. Export the model to ONNX: python serving/export_to_onnx.py
      2. Place the .onnx file at: serving/triton_model_config/finbert/1/model.onnx
      3. Start Triton: docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002
           -v $(pwd)/serving/triton_model_config:/models
           nvcr.io/nvidia/tritonserver:24.01-py3
           tritonserver --model-repository=/models
      4. Set TRITON_URL=localhost:8000 in your .env
    """

    # label order must match the FinBERT output head
    LABELS = ["positive", "negative", "neutral"]

    def __init__(self, url: str = None, model_name: str = "finbert"):
        self._url = url or os.getenv("TRITON_URL", "localhost:8000")
        self._model_name = model_name
        self._client = None
        self._fallback = None
        self._using_triton = False
        self._connect()

    def _connect(self):
        if not _HAVE_TRITON:
            self._init_fallback()
            return
        try:
            self._client = httpclient.InferenceServerClient(url=self._url, verbose=False)
            if self._client.is_server_live() and self._client.is_model_ready(self._model_name):
                self._using_triton = True
                logger.info("Connected to Triton at %s (model: %s)", self._url, self._model_name)
            else:
                logger.warning("Triton server not ready — falling back to local inference")
                self._init_fallback()
        except Exception as e:
            logger.info("Triton not reachable (%s) — falling back to local inference", e)
            self._init_fallback()

    def _init_fallback(self):
        try:
            from transformers import pipeline
            self._fallback = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=1,
                device=-1,  # CPU
            )
            logger.info("FinBERT fallback pipeline loaded")
        except Exception as e:
            logger.warning("Could not load FinBERT fallback: %s — FinBERT scoring disabled", e)
            self._fallback = None

    def classify(self, text: str) -> tuple:
        """
        Classify a text snippet as positive/negative/neutral.
        Returns (label, confidence_score).
        """
        if self._using_triton:
            return self._triton_classify(text)
        return self._local_classify(text)

    def classify_batch(self, texts: list) -> list:
        """Classify a list of texts, returns list of (label, score) tuples."""
        return [self.classify(t) for t in texts]

    # ── Triton path ────────────────────────────────────────────────────────────

    def _triton_classify(self, text: str) -> tuple:
        import numpy as np
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            enc = tok(text[:512], return_tensors="np", truncation=True, padding="max_length", max_length=128)

            inputs = []
            for name in ("input_ids", "attention_mask", "token_type_ids"):
                if name in enc:
                    t = httpclient.InferInput(name, enc[name].shape, "INT64")
                    t.set_data_from_numpy(enc[name].astype(np.int64))
                    inputs.append(t)

            outputs = [httpclient.InferRequestedOutput("logits")]
            result = self._client.infer(self._model_name, inputs, outputs=outputs)
            logits = result.as_numpy("logits")[0]
            # softmax
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
            idx = int(probs.argmax())
            return self.LABELS[idx], float(probs[idx])
        except Exception as e:
            logger.warning("Triton inference failed: %s — using fallback", e)
            return self._local_classify(text)

    # ── local fallback ─────────────────────────────────────────────────────────

    def _local_classify(self, text: str) -> tuple:
        if self._fallback is None:
            return "neutral", 0.0
        try:
            result = self._fallback(text[:512])
            if result and result[0]:
                top = result[0][0]
                return top.get("label", "neutral").lower(), float(top.get("score", 0.0))
        except Exception as e:
            logger.warning("Local FinBERT inference failed: %s", e)
        return "neutral", 0.0

    @property
    def is_using_triton(self) -> bool:
        return self._using_triton

    @property
    def backend(self) -> str:
        if self._using_triton:
            return "triton"
        if self._fallback is not None:
            return "local_finbert"
        return "disabled"
