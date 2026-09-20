"""Glowlytics ONNX models: skin signals + acne lesion detection."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
SKIN_SIGNALS_PATH = MODEL_DIR / "skin_signals.onnx"
ACNE_DETECTOR_PATH = MODEL_DIR / "acne_detector.onnx"

SIGNAL_NAMES = ("structure", "hydration", "sun_damage", "elasticity")
ACNE_CLASSES = ("comedone", "papule", "pustule", "nodule")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class GlowlyticsEngine:
    """Lazy-loaded ONNX inference for skin signals and acne boxes."""

    def __init__(self) -> None:
        self._signals: Any = None
        self._acne: Any = None
        self._available = False
        self._load()

    @property
    def available(self) -> bool:
        return self._available

    def _load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed; ML models disabled")
            return

        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 8
            opts.inter_op_num_threads = 4
            providers = ["CPUExecutionProvider"]
            if SKIN_SIGNALS_PATH.exists():
                self._signals = ort.InferenceSession(
                    str(SKIN_SIGNALS_PATH),
                    sess_options=opts,
                    providers=providers,
                )
            if ACNE_DETECTOR_PATH.exists():
                self._acne = ort.InferenceSession(
                    str(ACNE_DETECTOR_PATH),
                    sess_options=opts,
                    providers=providers,
                )
            self._available = self._signals is not None or self._acne is not None
            logger.info(
                "Glowlytics loaded signals=%s acne=%s",
                self._signals is not None,
                self._acne is not None,
            )
        except Exception:
            logger.exception("Failed to load Glowlytics models")
            self._available = False

    def predict_signals(self, img_bgr: np.ndarray) -> dict[str, float]:
        if self._signals is None:
            return {}

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        y0, x0 = (256 - 224) // 2, (256 - 224) // 2
        crop = resized[y0 : y0 + 224, x0 : x0 + 224].astype(np.float32) / 255.0
        crop = (crop - IMAGENET_MEAN) / IMAGENET_STD
        tensor = np.transpose(crop, (2, 0, 1))[None, ...].astype(np.float32)

        input_name = self._signals.get_inputs()[0].name
        out = self._signals.run(None, {input_name: tensor})[0][0]
        return {name: float(np.clip(out[i], 0.0, 1.0)) for i, name in enumerate(SIGNAL_NAMES)}

    def detect_acne(
        self,
        img_bgr: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> list[dict[str, Any]]:
        if self._acne is None:
            return []

        h0, w0 = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inp, ratio, pad = _letterbox(rgb, 640)
        tensor = inp.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        input_name = self._acne.get_inputs()[0].name
        raw = self._acne.run(None, {input_name: tensor})[0]
        return _parse_yolo(raw, conf_thres, iou_thres, ratio, pad, w0, h0)


_engine: Optional[GlowlyticsEngine] = None


def get_engine() -> GlowlyticsEngine:
    global _engine
    if _engine is None:
        _engine = GlowlyticsEngine()
    return _engine


def acne_heatmap(
    img_shape: tuple[int, ...],
    detections: list[dict[str, Any]],
) -> np.ndarray:
    h, w = img_shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], float(det["confidence"]))
    if heat.max() > 0:
        heat = cv2.GaussianBlur(heat, (31, 31), 0)
        heat = heat / heat.max()
    return heat


def _letterbox(img: np.ndarray, size: int = 640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, r, (left, top)


def _parse_yolo(
    raw: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    ratio: float,
    pad: tuple[int, int],
    orig_w: int,
    orig_h: int,
) -> list[dict[str, Any]]:
    pred = raw[0] if raw.ndim == 3 else raw
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    boxes = pred[:, :4]
    scores = pred[:, 4:]
    if scores.shape[1] == 0:
        return []

    class_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(len(scores)), class_ids]
    keep = confs >= conf_thres
    boxes, confs, class_ids = boxes[keep], confs[keep], class_ids[keep]
    if len(boxes) == 0:
        return []

    xyxy = _xywh_to_xyxy(boxes)
    pad_x, pad_y = pad
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / ratio
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / ratio
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_h)

    indices = _nms(xyxy, confs, iou_thres)
    detections: list[dict[str, Any]] = []
    for i in indices:
        cid = int(class_ids[i])
        label = ACNE_CLASSES[cid] if cid < len(ACNE_CLASSES) else f"class_{cid}"
        x1, y1, x2, y2 = xyxy[i].tolist()
        detections.append(
            {
                "label": label,
                "confidence": float(confs[i]),
                "box": [float(x1), float(y1), float(x2), float(y2)],
            }
        )
    return detections


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thres]
    return keep
