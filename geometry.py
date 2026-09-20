import cv2
import numpy as np

from config import BOX_EDGES, BOX_POINTS, CAMERA_MATRIX, DIST_COEFFICIENTS


def project_points(points_3d, rvec, tvec):
    projected, _ = cv2.projectPoints(
        points_3d, rvec, tvec, CAMERA_MATRIX, DIST_COEFFICIENTS
    )
    return projected


def draw_box(frame, rvec, tvec, color=(0, 0, 255)):
    points = project_points(BOX_POINTS, rvec, tvec).reshape(-1, 2).astype(int)
    for start, end in BOX_EDGES:
        cv2.line(frame, tuple(points[start]), tuple(points[end]), color, 2)
    return frame


def reprojection_errors(points_2d, points_3d, rvec, tvec):
    projected = project_points(points_3d, rvec, tvec).reshape(-1, 2)
    difference = points_2d.reshape(-1, 2) - projected
    return np.linalg.norm(difference, axis=1)


def mean_squared_reprojection_error(points_2d, points_3d, rvec, tvec):
    errors = reprojection_errors(points_2d, points_3d, rvec, tvec)
    return float(np.mean(errors ** 2))


def is_in_frame(point, height, width, margin=10):
    x, y = point.ravel()
    return margin < x < width - margin and margin < y < height - margin