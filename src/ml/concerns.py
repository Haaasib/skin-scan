"""Map scan scores to skincare concern tags for product matching."""
from __future__ import annotations

from typing import Any


ISSUE_CATALOG = {
    "acne": {
        "label": "Acne / breakouts",
        "tags": ["acne", "salicylic-acid", "benzoyl-peroxide", "niacinamide", "tea-tree"],
        "score_keys": ("acne", "blemishes", "acne_severity"),
    },
    "oiliness": {
        "label": "Excess oil",
        "tags": ["oily-skin", "oil-control", "niacinamide", "clay", "salicylic-acid"],
        "score_keys": ("oiliness", "oily_ml"),
    },
    "pores": {
        "label": "Visible pores",
        "tags": ["pores", "niacinamide", "retinol", "salicylic-acid"],
        "score_keys": ("pores", "structure_issue"),
    },
    "texture": {
        "label": "Uneven texture",
        "tags": ["texture", "exfoliant", "aha", "bha", "retinol"],
        "score_keys": ("texture", "structure_issue", "texture_trait"),
    },
    "dehydration": {
        "label": "Dehydration / dryness",
        "tags": ["hydration", "dry-skin", "hyaluronic-acid", "ceramide", "glycerin"],
        "score_keys": ("dehydration", "dry_ml"),
    },
    "redness": {
        "label": "Redness / sensitivity",
        "tags": ["redness", "sensitive", "centella", "panthenol", "soothing"],
        "score_keys": ("redness", "redness_trait"),
    },
    "pigmentation": {
        "label": "Pigmentation / dark spots",
        "tags": ["pigmentation", "dark-spots", "vitamin-c", "niacinamide", "arbutin"],
        "score_keys": ("pigment", "sun_damage", "dark_spots", "pigmentation_trait"),
    },
    "wrinkles": {
        "label": "Fine lines / wrinkles",
        "tags": ["anti-aging", "wrinkles", "retinol", "peptide", "bakuchiol"],
        "score_keys": ("wrinkles", "wrinkles_ml", "wrinkles_trait"),
    },
    "dark_circles": {
        "label": "Dark circles",
        "tags": ["dark-circles", "under-eye", "caffeine", "peptide", "vitamin-k"],
        "score_keys": ("dark_circles", "dark_circles_trait"),
    },
    "dullness": {
        "label": "Dull / low radiance",
        "tags": ["brightening", "radiance", "vitamin-c", "niacinamide", "exfoliant"],
        "score_keys": ("dullness",),
    },
    "elasticity": {
        "label": "Loss of firmness",
        "tags": ["anti-aging", "firming", "peptide", "retinol", "collagen"],
        "score_keys": ("elasticity_loss", "firmness_loss"),
    },
    "puffy_eyes": {
        "label": "Puffy under-eyes",
        "tags": ["puffiness", "under-eye", "caffeine", "cooling"],
        "score_keys": ("puffy_eyes",),
    },
}


def _severity(score: float) -> str:
    if score >= 0.65:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def build_issues(
    scores: dict[str, float],
    detections: list[dict[str, Any]] | None = None,
    skin_type: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[dict[str, Any]] = []
    tag_set: list[str] = []

    for issue_id, meta in ISSUE_CATALOG.items():
        vals = [scores[k] for k in meta["score_keys"] if k in scores]
        if not vals:
            continue
        score = float(max(vals))
        if score < 0.35:
            continue

        summary = f"{meta['label']} score {score * 100:.0f}%"
        if issue_id == "acne" and detections:
            counts: dict[str, int] = {}
            for det in detections:
                counts[det["label"]] = counts.get(det["label"], 0) + 1
            parts = [f"{n} {k}" for k, n in counts.items()]
            summary = f"{len(detections)} lesions detected ({', '.join(parts)})"

        issues.append(
            {
                "id": issue_id,
                "label": meta["label"],
                "severity": _severity(score),
                "score": round(score, 3),
                "summary": summary,
                "tags": meta["tags"],
            }
        )
        for t in meta["tags"]:
            if t not in tag_set:
                tag_set.append(t)

    if skin_type and skin_type != "unknown":
        st = skin_type.lower()
        if st not in tag_set:
            tag_set.insert(0, f"skin-type-{st}")
        if st == "oily" and "oily-skin" not in tag_set:
            tag_set.append("oily-skin")
        if st == "dry" and "dry-skin" not in tag_set:
            tag_set.append("dry-skin")

    issues.sort(key=lambda x: x["score"], reverse=True)
    return issues, tag_set
