"""Additional CV maps for full-face skincare scan."""
from __future__ import annotations

import cv2
import numpy as np


def _face_mask(masks: dict[str, np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    face = np.zeros(shape[:2], dtype=bool)
    for m in masks.values():
        face |= m > 0
    return face


def _normalize(heat: np.ndarray, face: np.ndarray) -> np.ndarray:
    out = heat.astype(np.float32)
    if face.any():
        vals = out[face]
        vmax = float(vals.max()) if vals.size else 0.0
        if vmax > 1e-6:
            out = out / vmax
    out = np.clip(out, 0.0, 1.0)
    out[~face] = 0
    return out.astype(np.float32)


def wrinkles_map(img_bgr: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    face = _face_mask(masks, img_bgr.shape)
    accum = np.zeros_like(gray)
    for theta in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        kern = cv2.getGaborKernel((15, 15), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern -= kern.mean()
        denom = np.abs(kern).sum() + 1e-6
        kern /= denom
        resp = np.abs(cv2.filter2D(gray, cv2.CV_32F, kern))
        accum = np.maximum(accum, resp)
    blur = cv2.GaussianBlur(accum, (5, 5), 0)
    region = masks.get("forehead", np.zeros(img_bgr.shape[:2], dtype=np.uint8))
    cheeks = masks.get("cheeks", np.zeros_like(region))
    focus = (region > 0) | (cheeks > 0)
    blur[~focus] *= 0.35
    return _normalize(blur, face)


def dark_circles_map(img_bgr: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    face = _face_mask(masks, img_bgr.shape)
    under = masks.get("under_eyes")
    cheeks = masks.get("cheeks")
    heat = np.zeros(img_bgr.shape[:2], dtype=np.float32)
    if under is None or cheeks is None or not (under > 0).any() or not (cheeks > 0).any():
        return heat
    cheek_L = float(np.median(L[cheeks > 0]))
    delta = np.clip(cheek_L - L, 0, None)
    heat[under > 0] = delta[under > 0]
    heat = cv2.GaussianBlur(heat, (21, 21), 0)
    return _normalize(heat, face)


def dark_spots_map(img_bgr: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    face = _face_mask(masks, img_bgr.shape)
    if not face.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)
    blur = cv2.GaussianBlur(L, (31, 31), 0)
    dark = blur - L
    dark = np.clip(dark, 0, None)
    _, binary = cv2.threshold(
        dark.astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    heat = cv2.GaussianBlur(binary.astype(np.float32), (11, 11), 0)
    return _normalize(heat, face)


def radiance_map(img_bgr: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    face = _face_mask(masks, img_bgr.shape)
    if not face.any():
        return np.zeros(img_bgr.shape[:2], dtype=np.float32)
    local = cv2.GaussianBlur(L, (15, 15), 0)
    uneven = np.abs(L - local)
    uneven = uneven / (uneven[face].max() + 1e-6)
    dull = 1.0 - np.clip(L / 255.0, 0, 1)
    issue = 0.55 * uneven + 0.45 * dull
    return _normalize(issue.astype(np.float32), face)


def firmness_map(img_bgr: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    face = _face_mask(masks, img_bgr.shape)
    edges = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edges = np.abs(edges)
    soft = cv2.GaussianBlur(edges, (9, 9), 0)
    soft = 1.0 - soft / (soft[face].max() + 1e-6) if face.any() else soft
    jaw = masks.get("chin", np.zeros(img_bgr.shape[:2], dtype=np.uint8))
    cheeks = masks.get("cheeks", np.zeros_like(jaw))
    heat = soft.copy()
    heat[(jaw == 0) & (cheeks == 0)] *= 0.25
    return _normalize(heat.astype(np.float32), face)
