"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel, Field


class IssueItem(BaseModel):
    id: str
    label: str
    severity: str
    score: float
    summary: str
    tags: list[str]


class DetectionItem(BaseModel):
    label: str
    confidence: float
    box: list[float]


class ScanResponse(BaseModel):
    """Response schema for skin scan analysis."""

    scores: dict[str, float]
    overlays: dict[str, str]
    regions: list[str]
    issues: list[IssueItem] = Field(default_factory=list)
    concern_tags: list[str] = Field(default_factory=list)
    detections: list[DetectionItem] = Field(default_factory=list)
    ml: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""

    ok: bool
    ml_loaded: bool = False
