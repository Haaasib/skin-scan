"""Pipeline composition - full multi-model skin scan."""
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
from .maps.extra import (
    wrinkles_map,
    dark_circles_map,
    dark_spots_map,
    radiance_map,
    firmness_map,
)
from .visualize import generate_all_overlays, draw_detections
from ..app.utils_io import encode_png_base64
from ..ml.glowlytics import acne_heatmap, get_engine
from ..ml.vit_panel import get_vit_panel
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


def _fuse_map(base: np.ndarray, strength: float) -> np.ndarray:
    return np.clip(base * 0.45 + float(strength), 0.0, 1.0).astype(np.float32)


def run_scan(img: np.ndarray) -> Dict[str, Any]:
    img_processed = preprocess(img, max_size=1280)

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
        "wrinkles": wrinkles_map(img_processed, masks),
        "dark_circles": dark_circles_map(img_processed, masks),
        "dark_spots": dark_spots_map(img_processed, masks),
        "dullness": radiance_map(img_processed, masks),
        "firmness_loss": firmness_map(img_processed, masks),
    }

    scores = {name: score_from_map(data, masks) for name, data in maps.items()}

    detections: list[dict[str, Any]] = []
    skin_type = None
    ml_meta: dict[str, Any] = {
        "glowlytics": False,
        "vit_panel": False,
        "stack": [],
    }

    engine = get_engine()
    if engine.available:
        ml_meta["glowlytics"] = True
        ml_meta["stack"].append("glowlytics")
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
            maps["hydration"] = _fuse_map(maps["hydration"], dehydration)

        if "sun_damage" in signals:
            scores["sun_damage"] = signals["sun_damage"]
            scores["pigment"] = _blend(scores["pigment"], signals["sun_damage"])
            maps["pigment"] = _fuse_map(maps["pigment"], signals["sun_damage"])

        if "elasticity" in signals:
            scores["elasticity_loss"] = 1.0 - signals["elasticity"]
            scores["firmness_loss"] = _blend(
                scores["firmness_loss"], scores["elasticity_loss"], 0.4, 0.6
            )

        acne_map = acne_heatmap(img_processed.shape, detections)
        if detections:
            lesion_score = float(
                min(
                    1.0,
                    0.25
                    + 0.08 * len(detections)
                    + 0.35
                    * float(np.mean([d["confidence"] for d in detections])),
                )
            )
        else:
            lesion_score = 0.0
        scores["acne"] = (
            _blend(scores.get("blemishes", 0.0), lesion_score, 0.35, 0.65)
            if detections
            else scores.get("blemishes", 0.0)
        )
        maps["acne"] = np.clip(
            np.maximum(maps["blemishes"] * 0.5, acne_map), 0.0, 1.0
        ).astype(np.float32)

    panel = get_vit_panel()
    if panel.available:
        ml_meta["vit_panel"] = True
        ml_meta["stack"].append("vit_panel")
        vit = panel.analyze(img_processed)
        ml_meta["vit"] = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in vit.items()
            if k != "traits"
        }
        if "traits" in vit:
            ml_meta["vit"]["traits"] = {k: round(v, 4) for k, v in vit["traits"].items()}

        if "acne_severity" in vit:
            scores["acne_severity"] = vit["acne_severity"]
            scores["acne"] = _blend(scores.get("acne", 0.0), vit["acne_severity"], 0.4, 0.6)
            scores["blemishes"] = _blend(scores["blemishes"], scores["acne"], 0.4, 0.6)

        if "wrinkles_ml" in vit:
            scores["wrinkles_ml"] = vit["wrinkles_ml"]
            scores["wrinkles"] = _blend(scores["wrinkles"], vit["wrinkles_ml"], 0.35, 0.65)
            maps["wrinkles"] = _fuse_map(maps["wrinkles"], vit["wrinkles_ml"])

        if "oily_ml" in vit:
            scores["oily_ml"] = vit["oily_ml"]
            scores["oiliness"] = _blend(scores["oiliness"], vit["oily_ml"], 0.4, 0.6)

        if "dry_ml" in vit:
            scores["dry_ml"] = vit["dry_ml"]
            scores["dehydration"] = _blend(
                scores.get("dehydration", scores["hydration"]), vit["dry_ml"], 0.4, 0.6
            )
            scores["hydration"] = scores["dehydration"]

        skin_type = vit.get("skin_type")
        ml_meta["skin_type"] = skin_type

        for key, val in vit.get("traits", {}).items():
            scores[key] = float(val)
            if key == "dark_circles_trait":
                scores["dark_circles"] = _blend(scores["dark_circles"], val, 0.4, 0.6)
                maps["dark_circles"] = _fuse_map(maps["dark_circles"], val)
            elif key == "pigmentation_trait":
                scores["pigment"] = _blend(scores["pigment"], val, 0.4, 0.6)
            elif key == "redness_trait":
                scores["redness"] = _blend(scores["redness"], val, 0.4, 0.6)
            elif key == "texture_trait":
                scores["texture"] = _blend(scores["texture"], val, 0.4, 0.6)
            elif key == "wrinkles_trait":
                scores["wrinkles"] = _blend(scores["wrinkles"], val, 0.4, 0.6)
            elif key == "puffy_eyes":
                scores["puffy_eyes"] = float(val)

    if "acne" not in scores:
        scores["acne"] = scores.get("blemishes", 0.0)
        maps["acne"] = maps["blemishes"]

    issues, concern_tags = build_issues(scores, detections, skin_type)

    overlay_images = generate_all_overlays(maps, alpha=0.6)
    overlays = {
        name: encode_png_base64(rgba) for name, rgba in overlay_images.items()
    }
    if detections:
        overlays["acne_boxes"] = encode_png_base64(
            draw_detections(img_processed, detections)
        )

    scores = {k: round(float(v), 4) for k, v in scores.items()}
    profile = {
        "skin_type": skin_type or "unknown",
        "top_issues": [i["id"] for i in issues[:5]],
        "models_used": ml_meta.get("stack", []),
    }

    return {
        "scores": scores,
        "overlays": overlays,
        "regions": list(masks.keys()),
        "issues": issues,
        "concern_tags": concern_tags,
        "detections": detections,
        "ml": ml_meta,
        "profile": profile,
    }
