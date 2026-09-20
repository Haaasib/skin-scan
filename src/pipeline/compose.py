"""Pipeline composition - orchestrates the full skin scan analysis."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .preprocess import preprocess
from .face_mesh import FaceMeshDetector, make_region_masks
from .maps.redness import redness_map
from .maps.oiliness import oiliness_map
from .maps.texture import texture_map
from .maps.pores import pores_map
from .maps.blemishes import blemish_map
from .maps.hydration import hydration_map
from .maps.pigment import pigment_map
from .visualize import generate_all_overlays, draw_detections
from ..app.utils_io import encode_png_base64
from ..ml.glowlytics import acne_heatmap, get_engine
from ..ml.concerns import build_issues


def score_from_map(map_data: np.ndarray, masks: dict[str, np.ndarray]) -> float:
    face_mask = np.zeros(map_data.shape, dtype=bool)
    for region_mask in masks.values():
        face_mask |= region_mask > 0
    if not face_mask.any():
        return 0.0
    return float(np.mean(map_data[face_mask]))


def _blend(a: float, b: float, wa: float = 0.45, wb: float = 0.55) -> float:
    return float(np.clip(a * wa + b * wb, 0.0, 1.0))


def run_scan(img: np.ndarray) -> Dict[str, Any]:
    img_processed = preprocess(img, max_size=1024)

    detector = FaceMeshDetector()
    landmarks = detector.detect(img_processed)
    if landmarks is None:
        raise ValueError("No face detected in image")

    masks = make_region_masks(landmarks, img_processed.shape)
    if not masks:
        raise ValueError("Could not create region masks from landmarks")

    maps = {
        "redness": redness_map(img_processed, masks),
        "oiliness": oiliness_map(img_processed, masks),
        "texture": texture_map(img_processed, masks),
        "pores": pores_map(img_processed, masks),
        "blemishes": blemish_map(img_processed, masks),
        "hydration": hydration_map(img_processed, masks),
        "pigment": pigment_map(img_processed, masks),
    }

    scores = {name: score_from_map(map_data, masks) for name, map_data in maps.items()}

    engine = get_engine()
    detections: list[dict[str, Any]] = []
    ml_meta: dict[str, Any] = {"engine": "glowlytics", "loaded": engine.available}

    if engine.available:
        signals = engine.predict_signals(img_processed)
        detections = engine.detect_acne(img_processed)
        ml_meta["signals_raw"] = {k: round(v, 4) for k, v in signals.items()}

        if "structure" in signals:
            structure_issue = 1.0 - signals["structure"]
            scores["structure_issue"] = structure_issue
            scores["texture"] = _blend(scores["texture"], structure_issue)
            scores["pores"] = _blend(scores["pores"], structure_issue)

        if "hydration" in signals:
            dehydration = 1.0 - signals["hydration"]
            scores["dehydration"] = dehydration
            scores["hydration"] = dehydration
            maps["hydration"] = np.clip(
                maps["hydration"] * 0.4 + dehydration, 0.0, 1.0
            ).astype(np.float32)

        if "sun_damage" in signals:
            scores["sun_damage"] = signals["sun_damage"]
            scores["pigment"] = _blend(scores["pigment"], signals["sun_damage"])
            maps["pigment"] = np.clip(
                maps["pigment"] * 0.4 + signals["sun_damage"], 0.0, 1.0
            ).astype(np.float32)

        if "elasticity" in signals:
            scores["elasticity_loss"] = 1.0 - signals["elasticity"]

        acne_map = acne_heatmap(img_processed.shape, detections)
        if detections:
            lesion_score = float(
                min(1.0, 0.25 + 0.08 * len(detections) + 0.35 * float(np.mean([d["confidence"] for d in detections])))
            )
        else:
            lesion_score = 0.0
        cv_blemish = scores.get("blemishes", 0.0)
        scores["acne"] = _blend(cv_blemish, lesion_score, 0.35, 0.65) if detections else cv_blemish
        maps["acne"] = np.clip(
            np.maximum(maps["blemishes"] * 0.5, acne_map), 0.0, 1.0
        ).astype(np.float32)
        if detections:
            scores["blemishes"] = _blend(scores["blemishes"], scores["acne"], 0.4, 0.6)

    issues, concern_tags = build_issues(scores, detections)

    overlay_images = generate_all_overlays(maps, alpha=0.6)
    overlays = {
        name: encode_png_base64(overlay_rgba)
        for name, overlay_rgba in overlay_images.items()
    }

    if detections:
        overlays["acne_boxes"] = encode_png_base64(draw_detections(img_processed, detections))

    scores = {k: round(float(v), 4) for k, v in scores.items()}

    return {
        "scores": scores,
        "overlays": overlays,
        "regions": list(masks.keys()),
        "issues": issues,
        "concern_tags": concern_tags,
        "detections": detections,
        "ml": ml_meta,
    }
