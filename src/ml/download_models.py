"""Download Glowlytics ONNX weights into src/ml/models."""
from __future__ import annotations

import urllib.request
from pathlib import Path

MODELS = {
    "skin_signals.onnx": (
        "https://huggingface.co/mufasabrownie/glowlytics-skin-models/resolve/main/skin_signals.onnx"
    ),
    "acne_detector.onnx": (
        "https://huggingface.co/mufasabrownie/glowlytics-skin-models/resolve/main/acne_detector.onnx"
    ),
}


def main() -> None:
    out_dir = Path(__file__).parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"skip {name}")
            continue
        print(f"download {name}")
        urllib.request.urlretrieve(url, dest)
        print(f"saved {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
