from dataclasses import dataclass

import cv2
import numpy as np


BOX_POINTS = np.array([
    [0, 90, 0], [125, 90, 0], [125, 90, 70], [0, 90, 70],
    [0, 0, 0], [125, 0, 0], [125, 0, 70], [0, 0, 70],
], dtype=np.float32).reshape((-1, 1, 3)) * 1e-3

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

CAMERA_MATRIX = np.array([
    [606.209, 0, 320.046],
    [0, 606.719, 238.926],
    [0, 0, 1],
], dtype=np.float32)
DIST_COEFFICIENTS = np.zeros((1, 5), dtype=np.float32)

POINTS_4 = (3, 2, 6, 7)
POINTS_6 = (3, 2, 6, 7, 0, 1)


@dataclass(frozen=True)
class TrackerConfig:
    backward_threshold: float = 1.0
    reprojection_threshold: float = 2.0
    exclusion_threshold: float = 3.0
    reintegration_threshold: float = 2.0
    texture_variance_min: float = 50.0
    texture_window: int = 15
    min_points_for_pnp: int = 4
    debug_from_frame: int = 300
    debug_point_threshold: float = 2.0
    lk_params: dict = None

    def __post_init__(self):
        if self.lk_params is None:
            object.__setattr__(self, "lk_params", {
                "winSize": (21, 21),
                "maxLevel": 3,
                "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            })