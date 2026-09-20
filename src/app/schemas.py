"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel, Field


class IssueItem(BaseModel):
    id: str = Field(..., description="Issue id, e.g. acne, dehydration")
    label: str = Field(..., description="Human-readable issue name")
    severity: str = Field(..., description="low | medium | high")
    score: float = Field(..., ge=0, le=1, description="Issue strength 0-1")
    summary: str = Field(..., description="Short explanation")
    tags: list[str] = Field(
        default_factory=list,
        description="Ingredient/concern tags for product matching",
    )


class DetectionItem(BaseModel):
    label: str = Field(..., description="Lesion class: comedone|papule|pustule|nodule")
    confidence: float = Field(..., ge=0, le=1)
    box: list[float] = Field(..., description="[x1,y1,x2,y2] pixel box")


class ScanResponse(BaseModel):
    """Full skin scan payload for product recommendation pipelines."""

    scores: dict[str, float] = Field(
        ...,
        description="All metric scores 0-1 (acne, oiliness, wrinkles, dark_circles, ...)",
    )
    overlays: dict[str, str] = Field(
        ...,
        description="PNG data-URIs for heatmaps (and acne_boxes if lesions found)",
    )
    regions: list[str] = Field(..., description="Detected face regions")
    issues: list[IssueItem] = Field(
        default_factory=list,
        description="Prioritized concerns with severity + product tags",
    )
    concern_tags: list[str] = Field(
        default_factory=list,
        description="Flat tag list to match against product ingredients",
    )
    detections: list[DetectionItem] = Field(
        default_factory=list,
        description="Acne lesion bounding boxes",
    )
    ml: dict = Field(default_factory=dict, description="Model diagnostics")
    profile: dict = Field(
        default_factory=dict,
        description="skin_type, top_issues, models_used",
    )


class HealthResponse(BaseModel):
    ok: bool
    ml_loaded: bool = False
    vit_loaded: bool = False
    models: list[str] = Field(default_factory=list)
