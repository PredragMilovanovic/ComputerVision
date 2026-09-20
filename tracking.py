import cv2
import numpy as np

from config import TrackerConfig


def track_forward_backward(gray_previous, gray_current, points, config=None):
    config = config or TrackerConfig()
    next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        gray_previous, gray_current, points, None, **config.lk_params
    )
    if next_points is None:
        count = len(points)
        return points.copy(), np.zeros(count, dtype=bool), np.full(count, np.inf)
    back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        gray_current, gray_previous, next_points, None, **config.lk_params
    )
    if back_points is None:
        count = len(points)
        return next_points, np.zeros(count, dtype=bool), np.full(count, np.inf)
    round_trip_error = np.linalg.norm(
        points.reshape(-1, 2) - back_points.reshape(-1, 2), axis=1
    )
    valid = (
        (forward_status.ravel() == 1)
        & (backward_status.ravel() == 1)
        & (round_trip_error < config.backward_threshold)
    )
    return next_points, valid, round_trip_error


def local_texture_variance(gray, points, window=15):
    height, width = gray.shape
    variances = np.zeros(len(points), dtype=np.float32)
    for index, (x, y) in enumerate(points.astype(int)):
        x0, x1 = max(0, x - window), min(width, x + window)
        y0, y1 = max(0, y - window), min(height, y + window)
        patch = gray[y0:y1, x0:x1]
        if patch.size:
            variances[index] = np.var(patch)
    return variances