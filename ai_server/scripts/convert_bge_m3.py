#!/usr/bin/env python3
"""BGE-M3 모델을 ONNX로 변환 (Hugging Face Optimum)"""

from pathlib import Path

MODEL_NAME = "BAAI/bge-m3"
OUTPUT_DIR = "/opt/models/bge-m3-onnx"


def convert_model():
    output_path = Path(OUTPUT_DIR)

    # 이미 변환된 경우 스킵
    if output_path.exists() and (output_path / "model.onnx").exists():
        print(f"Model already exists at {OUTPUT_DIR}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Converting {MODEL_NAME} to ONNX...")

    # Optimum Python API 사용 (CLI 대신)
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    # 모델 로드 및 ONNX 변환
    model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_NAME,
        export=True,  # PyTorch → ONNX 변환
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 저장
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_model()
