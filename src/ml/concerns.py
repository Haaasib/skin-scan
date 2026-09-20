"""Map scan scores to skincare concern tags for product matching."""
from __future__ import annotations

from typing import Any


ISSUE_CATALOG = {
    "acne": {
        "label": "Acne / breakouts",
        "tags": ["acne", "salicylic-acid", "benzoyl-peroxide", "niacinamide", "tea-tree"],
        "score_keys": ("acne", "blemishes"),
    },
    "oiliness": {
        "label": "Excess oil",
        "tags": ["oily-skin", "oil-control", "niacinamide", "clay", "salicylic-acid"],
        "score_keys": ("oiliness",),
    },
    "pores": {
        "label": "Visible pores",
        "tags": ["pores", "niacinamide", "retinol", "salicylic-acid"],
        "score_keys": ("pores", "structure_issue"),
    },
    "texture": {
        "label": "Uneven texture",
        "tags": ["texture", "exfoliant", "aha", "bha", "retinol"],
        "score_keys": ("texture", "structure_issue"),
    },
    "dehydration": {
        "label": "Dehydration / dryness",
        "tags": ["hydration", "dry-skin", "hyaluronic-acid", "ceramide", "glycerin"],
        "score_keys": ("dehydration",),
    },
    "redness": {
        "label": "Redness / sensitivity",
        "tags": ["redness", "sensitive", "centella", "panthenol", "soothing"],
        "score_keys": ("redness",),
    },
    "pigmentation": {
        "label": "Pigmentation / dark spots",
        "tags": ["pigmentation", "dark-spots", "vitamin-c", "niacinamide", "arbutin"],
        "score_keys": ("pigment", "sun_damage"),
    },
    "elasticity": {
        "label": "Loss of firmness",
        "tags": ["anti-aging", "firming", "peptide", "retinol", "collagen"],
        "score_keys": ("elasticity_loss",),
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

    issues.sort(key=lambda x: x["score"], reverse=True)
    return issues, tag_set
