from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import cv2

from config import BOX_POINTS, CAMERA_MATRIX, DIST_COEFFICIENTS, TrackerConfig
from geometry import draw_box, is_in_frame, mean_squared_reprojection_error, project_points, reprojection_errors
from tracking import local_texture_variance, track_forward_backward
from visualization import draw_points, save_debug_frame


def select_points(frame, count, labels, window_name):
    clicked = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < count:
            clicked.append((x, y))
            preview = frame.copy()
            for index, point in enumerate(clicked):
                cv2.circle(preview, point, 7, (0, 255, 0), -1)
                cv2.putText(preview, f"P{labels[index]}", (point[0] + 9, point[1] - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, preview)

    cv2.imshow(window_name, frame)
    cv2.setMouseCallback(window_name, on_click)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    if len(clicked) < 4:
        raise RuntimeError(f"Not enough points selected ({len(clicked)}/{count}).")
    return clicked[:count]


def _solve_pnp(object_points, image_points, rvec=None, tvec=None, use_extrinsic_guess=False):
    point_count = len(object_points)
    if point_count < 4:
        return False, None, None

    solver = cv2.SOLVEPNP_ITERATIVE
    try:
        return cv2.solvePnP(
            object_points, image_points, CAMERA_MATRIX, DIST_COEFFICIENTS,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=use_extrinsic_guess,
            flags=solver,
        )
    except cv2.error:
        if point_count >= 6 or not hasattr(cv2, "SOLVEPNP_SQPNP"):
            return False, None, None
        try:
            return cv2.solvePnP(
                object_points, image_points, CAMERA_MATRIX, DIST_COEFFICIENTS,
                rvec=rvec, tvec=tvec, useExtrinsicGuess=False,
                flags=cv2.SOLVEPNP_SQPNP,
            )
        except cv2.error:
            return False, None, None


def run_pipeline(video_path, point_indices, label, output_dir=".", debug=False,
                 initial_points=None, display=True, max_frames=None, config=None):
    config = config or TrackerConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise IOError(f"Cannot read first frame from: {video_path}")

    height, width = first_frame.shape[:2]
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    gray_previous = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    count = len(point_indices)
    labels = [str(index + 1) for index in point_indices]
    clicked = initial_points or select_points(first_frame, count, labels, f"Init [{label}] - click points, then press ENTER")
    points_previous = np.asarray(clicked, dtype=np.float32).reshape(count, 1, 2)
    tracked_object_points = BOX_POINTS[list(point_indices)]

    debug_dir = output_dir / f"debug_frames_{label}"
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        init_frame = first_frame.copy()
        draw_points(init_frame, points_previous, labels, np.ones(count, bool), np.ones(count, bool),
                    np.full(count, np.nan), np.zeros(count, bool), config)
        cv2.imwrite(str(debug_dir / "frame_0000_init.jpg"), init_frame)

    output_path = output_dir / f"augmented_{label}.avi"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
    mse_values = []
    previous_rvec, previous_tvec = None, None
    excluded = np.zeros(count, dtype=bool)
    frame_index = 0
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            points_next, klt_ok, round_trip_error = track_forward_backward(
                gray_previous, gray_current, points_previous, config
            )
            texture_variance = local_texture_variance(gray_current, points_next.reshape(-1, 2), config.texture_window)
            texture_ok = texture_variance >= config.texture_variance_min
            in_frame = np.array([is_in_frame(point, height, width) for point in points_next])
            usable = klt_ok & texture_ok & in_frame & ~excluded
            reprojection_error = np.full(count, np.nan)
            current_mse = None

            if int(usable.sum()) >= config.min_points_for_pnp:
                success, rvec, tvec = _solve_pnp(
                    tracked_object_points[usable], points_next[usable],
                    rvec=previous_rvec, tvec=previous_tvec,
                    use_extrinsic_guess=previous_rvec is not None,
                )
                if success:
                    reprojection_error = reprojection_errors(
                        points_next.reshape(count, 2), tracked_object_points.reshape(count, 3), rvec, tvec
                    )
                    for index in range(count):
                        if excluded[index]:
                            projected = project_points(tracked_object_points[index], rvec, tvec)
                            if is_in_frame(projected, height, width) and np.linalg.norm(points_next[index] - projected) < config.reintegration_threshold:
                                excluded[index] = False
                                points_next[index] = projected
                        elif reprojection_error[index] > config.exclusion_threshold:
                            excluded[index] = True
                    active = ~excluded & in_frame
                    if int(active.sum()) >= config.min_points_for_pnp:
                        success, refined_rvec, refined_tvec = _solve_pnp(
                            tracked_object_points[active], points_next[active],
                            rvec=rvec, tvec=tvec, use_extrinsic_guess=True,
                        )
                        if success:
                            rvec, tvec = refined_rvec, refined_tvec
                            reprojection_error = reprojection_errors(
                                points_next.reshape(count, 2), tracked_object_points.reshape(count, 3), rvec, tvec
                            )
                    previous_rvec, previous_tvec = rvec.copy(), tvec.copy()
                    frame = draw_box(frame, rvec, tvec)
                    active = ~excluded & in_frame
                    if int(active.sum()) >= config.min_points_for_pnp:
                        current_mse = mean_squared_reprojection_error(
                            points_next[active], tracked_object_points[active], rvec, tvec
                        )
                        mse_values.append(current_mse)
                        cv2.putText(frame, f"[{label}] f={frame_index} active={active.sum()}/{count} MSE={current_mse:.3f}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            draw_points(frame, points_next, labels, klt_ok, texture_ok, reprojection_error, excluded, config)
            if debug and (frame_index >= config.debug_from_frame or np.nanmax(reprojection_error) > config.debug_point_threshold):
                save_debug_frame(debug_dir / f"frame_{frame_index:04d}.jpg", frame.copy(), frame_index, labels,
                                 round_trip_error, texture_variance, texture_ok, reprojection_error,
                                 excluded, klt_ok, current_mse, config)
            writer.write(frame)
            if display:
                cv2.imshow(f"Augmented [{label}]", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            gray_previous, points_previous = gray_current, points_next
            frame_index += 1
    finally:
        capture.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()
    return mse_values, frame_index


def run_sensitivity_analysis(video_path, point_indices, base_click_points, label,
                             n_trials=5, jitter_px=1):
    """Measure mean MSE variation caused by jittering the initial click points."""
    trial_means = []
    base_points = np.asarray(base_click_points, dtype=np.float32)
    with TemporaryDirectory() as output_dir:
        for trial_index in range(n_trials):
            jitter = np.random.uniform(-jitter_px, jitter_px, size=base_points.shape)
            initial_points = base_points + jitter
            mse_values, _ = run_pipeline(
                video_path,
                point_indices,
                f"{label}_sensitivity_{trial_index + 1}",
                output_dir=output_dir,
                initial_points=initial_points.tolist(),
                display=False,
            )
            trial_means.append(float(np.mean(mse_values)))
    return trial_means