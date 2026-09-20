"""Visualization utilities for creating heatmap overlays."""
from typing import Optional
import cv2
import numpy as np


def create_heatmap_overlay(
    map_data: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Create RGBA heatmap overlay from normalized map data.

    Args:
        map_data: Normalized map [0, 1] as float32
        colormap: OpenCV colormap (default: COLORMAP_JET)
        alpha: Transparency of heatmap (0=transparent, 1=opaque)

    Returns:
        RGBA image with heatmap overlay
    """
    # Convert to 8-bit
    map_uint8 = (map_data * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(map_uint8, colormap)

    # Convert to RGBA
    heatmap_rgba = cv2.cvtColor(heatmap, cv2.COLOR_BGR2BGRA)

    # Set alpha channel based on map intensity and alpha parameter
    # Areas with no data (0) should be fully transparent
    alpha_channel = (map_data * alpha * 255).astype(np.uint8)
    heatmap_rgba[..., 3] = alpha_channel

    return heatmap_rgba


def overlay_on_image(
    base_img: np.ndarray,
    overlay_rgba: np.ndarray
) -> np.ndarray:
    """
    Composite RGBA overlay onto base BGR image.

    Args:
        base_img: Base BGR image
        overlay_rgba: RGBA overlay

    Returns:
        Composited BGR image
    """
    # Ensure same size
    if base_img.shape[:2] != overlay_rgba.shape[:2]:
        overlay_rgba = cv2.resize(
            overlay_rgba,
            (base_img.shape[1], base_img.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    # Convert base to BGRA
    if base_img.shape[2] == 3:
        base_bgra = cv2.cvtColor(base_img, cv2.COLOR_BGR2BGRA)
    else:
        base_bgra = base_img.copy()

    # Extract alpha
    alpha = overlay_rgba[..., 3:4].astype(np.float32) / 255.0

    # Blend
    overlay_bgr = overlay_rgba[..., :3].astype(np.float32)
    base_bgr = base_bgra[..., :3].astype(np.float32)

    blended = overlay_bgr * alpha + base_bgr * (1 - alpha)
    blended = blended.astype(np.uint8)

    # Set alpha to opaque
    result = cv2.cvtColor(blended, cv2.COLOR_BGR2BGRA)
    result[..., 3] = 255

    return cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)


def generate_all_overlays(
    maps: dict[str, np.ndarray],
    colormaps: Optional[dict[str, int]] = None,
    alpha: float = 0.6
) -> dict[str, np.ndarray]:
    """
    Generate RGBA overlays for all maps.

    Args:
        maps: Dict of normalized maps
        colormaps: Optional dict mapping map names to colormap constants
        alpha: Transparency

    Returns:
        Dict of RGBA overlays
    """
    if colormaps is None:
        # Default colormaps per category
        colormaps = {
            "redness": cv2.COLORMAP_HOT,
            "oiliness": cv2.COLORMAP_VIRIDIS,
            "texture": cv2.COLORMAP_BONE,
            "pores": cv2.COLORMAP_COOL,
            "blemishes": cv2.COLORMAP_AUTUMN,
            "acne": cv2.COLORMAP_HOT,
            "hydration": cv2.COLORMAP_OCEAN,
            "pigment": cv2.COLORMAP_PINK,
        }

    overlays = {}
    for name, map_data in maps.items():
        colormap = colormaps.get(name, cv2.COLORMAP_JET)
        overlays[name] = create_heatmap_overlay(map_data, colormap, alpha)

    return overlays


def draw_detections(
    img_bgr: np.ndarray,
    detections: list,
) -> np.ndarray:
    out = img_bgr.copy()
    colors = {
        "comedone": (80, 180, 255),
        "papule": (0, 140, 255),
        "pustule": (0, 80, 255),
        "nodule": (0, 0, 220),
    }
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        color = colors.get(det["label"], (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det['label']} {det['confidence']:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out
