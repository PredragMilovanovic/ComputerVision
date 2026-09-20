import cv2
import numpy as np

from config import TrackerConfig


def draw_points(frame, points, labels, klt_ok, texture_ok, reprojection_error, excluded, config=None):
    config = config or TrackerConfig()
    for index, point in enumerate(points.reshape(-1, 2).astype(int)):
        error = reprojection_error[index]
        if not np.isnan(error) and error >= config.reprojection_threshold:
            color = (0, 0, 220)
        elif excluded[index]:
            color = (0, 0, 180)
        elif not klt_ok[index]:
            color = (0, 165, 255)
        elif not texture_ok[index]:
            color = (0, 215, 255)
        else:
            color = (0, 220, 0)
        cv2.circle(frame, tuple(point), 7, color, -1)
        cv2.circle(frame, tuple(point), 7, (255, 255, 255), 1)
        value = f"{error:.2f}" if not np.isnan(error) else "?"
        text_color = (0, 0, 220) if value != "?" and error >= config.reprojection_threshold else color
        cv2.putText(frame, f"P{labels[index]} rp={value}", (point[0] + 9, point[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)


def save_debug_frame(path, frame, frame_index, labels, round_trip_error,
                     texture_variance, texture_ok, reprojection_error,
                     excluded, klt_ok, current_mse, config=None):
    config = config or TrackerConfig()
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 125), (width, height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"Frame {frame_index:04d}", (10, height - 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    mse_text = f"MSE = {current_mse:.4f} px2" if current_mse is not None else "MSE = N/A"
    cv2.putText(frame, mse_text, (160, height - 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 1)
    for index, label in enumerate(labels):
        error = reprojection_error[index]
        error_text = f"{error:.2f}" if not np.isnan(error) else "--"
        if excluded[index]:
            status, color = "EXCL", (80, 80, 220)
        elif not klt_ok[index]:
            status, color = "KLT?", (0, 130, 255)
        elif not texture_ok[index]:
            status, color = "TEX?", (0, 200, 255)
        elif not np.isnan(error) and error > config.exclusion_threshold:
            status, color = "OUT", (0, 80, 255)
        else:
            status, color = "ok", (0, 210, 0)
        text = f"P{label}: fb={round_trip_error[index]:.2f} rp={error_text} tex={texture_variance[index]:.0f} [{status}]"
        cv2.putText(frame, text, (10 + (index % 3) * 215, height - 75 + (index // 3) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, color, 1)
    cv2.imwrite(str(path), frame)