"""HuggingFace ViT panel for full-face skin classification."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

ACNE_SEVERITY = {
    "level_-1": 0.0,
    "clear": 0.0,
    "level_0": 0.2,
    "occasional": 0.2,
    "level_1": 0.4,
    "mild": 0.4,
    "level_2": 0.6,
    "moderate": 0.6,
    "level_3": 0.8,
    "severe": 0.8,
    "level_4": 1.0,
    "very": 1.0,
}

TRAIT_TO_SCORE = {
    "acne": "acne_trait",
    "eyes dark circles": "dark_circles_trait",
    "puffy eyes": "puffy_eyes",
    "skin pigmentation": "pigmentation_trait",
    "skin redness": "redness_trait",
    "skin texture": "texture_trait",
    "wrinkled face": "wrinkles_trait",
}


class VitSkinPanel:
    def __init__(self) -> None:
        self._pipes: dict[str, Any] = {}
        self._available = False
        self._load()

    @property
    def available(self) -> bool:
        return self._available

    def _load(self) -> None:
        try:
            import torch
            from transformers import pipeline
        except ImportError:
            logger.warning("torch/transformers missing; ViT panel disabled")
            return

        torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
        device = 0 if torch.cuda.is_available() else -1
        specs = {
            "acne": "imfarzanansari/skintelligent-acne",
            "wrinkles": "imfarzanansari/skintelligent-wrinkles",
            "skin_type": "dima806/skin_types_image_detection",
            "traits": "varun1505/face-characteristics",
        }
        for key, model_id in specs.items():
            try:
                self._pipes[key] = pipeline(
                    "image-classification",
                    model=model_id,
                    device=device,
                )
                logger.info("Loaded ViT model %s", model_id)
            except Exception:
                logger.exception("Failed to load %s", model_id)

        self._available = bool(self._pipes)

    def analyze(self, img_bgr: np.ndarray) -> dict[str, Any]:
        if not self._available:
            return {}

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        out: dict[str, Any] = {"models": list(self._pipes.keys())}

        if "acne" in self._pipes:
            preds = self._pipes["acne"](pil, top_k=None)
            out["acne_severity"] = _map_acne(preds)

        if "wrinkles" in self._pipes:
            preds = self._pipes["wrinkles"](pil, top_k=None)
            out["wrinkles_ml"] = _positive_score(preds, positive_tokens=("wrinkle", "yes", "1"))

        if "skin_type" in self._pipes:
            preds = self._pipes["skin_type"](pil, top_k=None)
            out["skin_type"] = _top_label(preds)
            out["oily_ml"] = _label_score(preds, "oily")
            out["dry_ml"] = _label_score(preds, "dry")

        if "traits" in self._pipes:
            preds = self._pipes["traits"](pil, top_k=None)
            traits = {}
            for p in preds:
                label = str(p["label"]).strip().lower()
                key = TRAIT_TO_SCORE.get(label)
                if key:
                    traits[key] = float(p["score"])
            out["traits"] = traits

        return out


_panel: Optional[VitSkinPanel] = None


def get_vit_panel() -> VitSkinPanel:
    global _panel
    if _panel is None:
        _panel = VitSkinPanel()
    return _panel


def _top_label(preds: list[dict]) -> str:
    if not preds:
        return "unknown"
    best = max(preds, key=lambda x: float(x["score"]))
    return str(best["label"]).lower()


def _label_score(preds: list[dict], needle: str) -> float:
    needle = needle.lower()
    for p in preds:
        if needle in str(p["label"]).lower():
            return float(p["score"])
    return 0.0


def _positive_score(preds: list[dict], positive_tokens: tuple[str, ...]) -> float:
    best = 0.0
    for p in preds:
        label = str(p["label"]).lower()
        if any(t in label for t in positive_tokens):
            best = max(best, float(p["score"]))
        if label in ("0", "no", "none", "non-wrinkle", "clear"):
            best = max(best, 1.0 - float(p["score"]))
    if best == 0.0 and preds:
        ranked = sorted(preds, key=lambda x: float(x["score"]), reverse=True)
        top = ranked[0]
        label = str(top["label"]).lower()
        if any(t in label for t in ("wrinkle", "yes", "positive", "1")):
            return float(top["score"])
        return 1.0 - float(top["score"])
    return float(np.clip(best, 0.0, 1.0))


def _map_acne(preds: list[dict]) -> float:
    if not preds:
        return 0.0
    score = 0.0
    for p in preds:
        label = str(p["label"]).lower().replace(" ", "_").replace("-", "_")
        weight = 0.5
        for key, val in ACNE_SEVERITY.items():
            if key in label:
                weight = val
                break
        digits = "".join(ch if ch.isdigit() or ch == "-" else " " for ch in label).split()
        for d in digits:
            try:
                lvl = int(d)
                if lvl == -1:
                    weight = 0.0
                elif lvl == 0:
                    weight = 0.2
                elif lvl == 1:
                    weight = 0.4
                elif lvl == 2:
                    weight = 0.6
                elif lvl == 3:
                    weight = 0.8
                elif lvl >= 4:
                    weight = 1.0
            except ValueError:
                pass
        score += weight * float(p["score"])
    return float(np.clip(score, 0.0, 1.0))
