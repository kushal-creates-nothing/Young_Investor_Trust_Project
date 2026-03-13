"""
Export ProsusAI/finbert to ONNX format for Triton Inference Server deployment.

Run once after fine-tuning or when you want to update the served model:
    python serving/export_to_onnx.py [--model-path <path>]

The output goes to serving/triton_model_config/finbert/1/model.onnx
"""

import argparse
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def export(model_path: str = "ProsusAI/finbert", output_dir: str = None):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        logger.error("transformers and torch are required for ONNX export")
        return False

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "triton_model_config/finbert/1",
        )
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "model.onnx")

    logger.info("Loading model from: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # dummy input for tracing
    sample = "The stock market rallied strongly today on positive economic data."
    enc = tokenizer(sample, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    logger.info("Exporting to ONNX: %s", onnx_path)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]),
            onnx_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
            opset_version=14,
        )
    logger.info("ONNX export complete: %s (%.1f MB)", onnx_path, os.path.getsize(onnx_path) / 1e6)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="ProsusAI/finbert", help="HuggingFace model path or local dir")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    success = export(args.model_path, args.output_dir)
    exit(0 if success else 1)
