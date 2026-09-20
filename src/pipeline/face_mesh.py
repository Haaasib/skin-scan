"""Face mesh detection using MediaPipe and region mask generation."""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional


class FaceMeshDetector:
    """MediaPipe face mesh detector."""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def detect(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face landmarks.

        Args:
            img_bgr: Input BGR image

        Returns:
            Landmarks as (N, 2) array of (x, y) pixel coordinates, or None
        """
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return None

        # Get first face
        face_landmarks = results.multi_face_landmarks[0]

        # Convert normalized coordinates to pixels
        h, w = img_bgr.shape[:2]
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])

        return np.array(landmarks, dtype=np.int32)

    def __del__(self):
        if hasattr(self, "face_mesh"):
            self.face_mesh.close()


def make_region_masks(landmarks: np.ndarray, img_shape: tuple) -> dict[str, np.ndarray]:
    """
    Create region masks for different facial areas.

    Regions: forehead, cheeks (left/right), nose, chin

    Args:
        landmarks: (N, 2) array of landmark coordinates
        img_shape: (height, width) of image

    Returns:
        Dict mapping region name to binary mask
    """
    h, w = img_shape[:2]
    masks = {}

    # MediaPipe face mesh landmark indices for regions
    # These are approximate and can be tuned

    # Forehead (approximate using face oval top)
    forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    # Left cheek
    left_cheek_indices = [234, 93, 132, 58, 172, 136, 150, 176, 148, 152,
                          377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
                          356, 389, 251, 284, 332, 297, 338]

    # Right cheek (mirror of left)
    right_cheek_indices = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                           152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
                           234, 127, 162, 21, 54, 103, 67, 109]

    # Nose
    nose_indices = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12,
                    13, 14, 15, 16, 17, 18, 200, 199, 175, 152]

    # Chin
    chin_indices = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
                    356, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54,
                    21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 176]

    # Helper function to create mask from indices
    def create_mask_from_indices(indices, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        if len(indices) == 0:
            return mask

        # Get valid landmarks
        valid_indices = [i for i in indices if i < len(landmarks)]
        if len(valid_indices) < 3:
            return mask

        points = landmarks[valid_indices]

        # Create convex hull
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

        return mask

    # Create all region masks
    masks["forehead"] = create_mask_from_indices(forehead_indices[:20], (h, w))  # Use subset
    masks["nose"] = create_mask_from_indices(nose_indices, (h, w))
    masks["chin"] = create_mask_from_indices([152, 175, 200, 201, 18, 421, 406, 335, 273], (h, w))

    # For cheeks, use simple approach with full face landmarks
    # Left cheek: left side of face
    left_side = landmarks[landmarks[:, 0] < w // 2]
    if len(left_side) > 3:
        hull = cv2.convexHull(left_side)
        masks["left_cheek"] = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(masks["left_cheek"], hull, 255)

    # Right cheek: right side of face
    right_side = landmarks[landmarks[:, 0] >= w // 2]
    if len(right_side) > 3:
        hull = cv2.convexHull(right_side)
        masks["right_cheek"] = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(masks["right_cheek"], hull, 255)

    # Combine cheeks into one "cheeks" region
    if "left_cheek" in masks and "right_cheek" in masks:
        masks["cheeks"] = cv2.bitwise_or(masks["left_cheek"], masks["right_cheek"])
        del masks["left_cheek"]
        del masks["right_cheek"]

    under_eye_idx = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,
                     362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
    under = create_mask_from_indices(under_eye_idx, (h, w))
    if under.any():
        under = cv2.dilate(under, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        shift = np.zeros_like(under)
        shift[8:, :] = under[:-8, :]
        masks["under_eyes"] = shift

    return masks
