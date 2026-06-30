"""Distraction Monitoring System (DMS)
Real-time driver attention monitoring using MediaPipe Face Landmarker.
"""

import cv2
import time
import os
import sys
import urllib.request
from collections import deque

import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Settings for detection and UI
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

OUTPUT_VIDEO_NAME = "DMS_output.avi"

MIN_CALIBRATION_SAMPLES = 60

LONG_DISTRACTION_TIME = 5.0        # long distraction limit
SHORT_DISTRACTION_WINDOW = 30.0    # window for short-distraction totals
SHORT_DISTRACTION_TOTAL = 10.0     # short-distraction limit inside the window
FOCUSED_RESET_TIME = 2.0           # time needed to update baselines

MICROSLEEP_TIME = 4.0              # microsleep limit with eyes closed
SLEEP_TIME = 7.0                   # sleep limit with eyes closed
BLINK_GRACE_TIME = 0.40            # ignore blinks shorter than this

# Smoothing factors for noisy measurements
GAZE_EMA_ALPHA = 0.35
IRIS_EMA_ALPHA = 0.30

# Face-loss handling
NO_FACE_AS_DISTRACTION = True
NO_FACE_GRACE_TIME = 0.5

# EAR smoothing when opening or closing
EAR_EMA_ALPHA_OPEN = 0.35
EAR_EMA_ALPHA_CLOSE = 0.80

# Lizard tuning: eyes move while the head stays forward.
# Lower values make it easier to trigger.
LIZARD_HEAD_FORWARD_MULT = 1.25

# Debug flags
SHOW_LANDMARK_VISUALIZATION = False
SHOW_DEBUG_TEXT = False
SHOW_RPPG_ROI = False

RPPG_WINDOW_SECONDS = 10.0
RPPG_MIN_BPM = 45
RPPG_MAX_BPM = 180

# UI constants
UI_BOTTOM_BAR_HEIGHT = 55
UI_BOTTOM_FONT_SCALE = 0.60
UI_BOTTOM_THICKNESS = 2

UI_DEBUG_FONT_SCALE = 0.50
UI_DEBUG_THICKNESS = 1


# MediaPipe landmark indices
NOSE_TIP = 1

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

LEFT_EYE_VIS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_VIS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

LEFT_IRIS = [473, 474, 475, 476, 477]
RIGHT_IRIS = [468, 469, 470, 471, 472]

NOSE_VIS = [45, 4, 275, 1]

FACE_LEFT = 234
FACE_RIGHT = 454
CHIN = 152
FOREHEAD = 10


# Helper functions

def download_model():
    # Download the MediaPipe task file if it is missing.
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def get_landmark_px(landmarks, index, width, height):
    lm = landmarks[index]
    return np.array([int(lm.x * width), int(lm.y * height)])


def draw_landmark_visualization(frame, face_landmarks, width, height):
    # Draw landmarks only when debug view is enabled.
    if not SHOW_LANDMARK_VISUALIZATION:
        return

    for idx, lm in enumerate(face_landmarks):
        x = int(lm.x * width)
        y = int(lm.y * height)

        if idx in LEFT_EYE_VIS or idx in RIGHT_EYE_VIS:
            # eye outline points
            cv2.circle(frame, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

        if idx in LEFT_IRIS or idx in RIGHT_IRIS:
            # iris points
            cv2.circle(frame, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

        if idx in NOSE_VIS:
            # nose points
            cv2.circle(frame, (x, y), radius=3, color=(255, 0, 0), thickness=-1)


def eye_aspect_ratio(landmarks, width, height, top, bottom, left, right):
    top_p = get_landmark_px(landmarks, top, width, height)
    bottom_p = get_landmark_px(landmarks, bottom, width, height)
    left_p = get_landmark_px(landmarks, left, width, height)
    right_p = get_landmark_px(landmarks, right, width, height)

    vertical = np.linalg.norm(top_p - bottom_p)
    horizontal = np.linalg.norm(left_p - right_p)

    # Avoid division by zero.
    if horizontal <= 1e-6:
        return 0.0

    return vertical / horizontal


def fastica(X, max_iter=200, tol=1e-4):
    # Small local FastICA implementation.
    # Center, whiten, then run fixed-point updates.
    X = X - np.mean(X, axis=1, keepdims=True)

    cov = np.cov(X)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    eig_vals[eig_vals < 1e-8] = 1e-8

    # Whitening step.
    whitening = np.diag(1.0 / np.sqrt(eig_vals)) @ eig_vecs.T
    Z = whitening @ X

    n_components, n_samples = Z.shape
    W = np.zeros((n_components, n_components))

    # Estimate one component at a time.
    for p in range(n_components):
        w = np.random.randn(n_components)
        w = w / np.linalg.norm(w)

        for _ in range(max_iter):
            w_old = w.copy()

            y = w.T @ Z
            g = np.tanh(y)
            gp = 1.0 - g ** 2

            # Fixed-point update.
            w = (Z @ g.T) / n_samples - np.mean(gp) * w

            # Remove correlation with earlier components.
            if p > 0:
                w = w - W[:p].T @ (W[:p] @ w)

            norm_w = np.linalg.norm(w)
            if norm_w <= 1e-8:
                break

            w = w / norm_w

            # Stop when the vector stops changing.
            if abs(abs(np.dot(w, w_old)) - 1.0) < tol:
                break

        W[p, :] = w

    return W @ Z


def robust_std(values):
    # Robust standard deviation estimate using MAD.
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return 1.4826 * mad


def estimate_bpm_from_rgb(rgb_buffer, time_buffer):
    # Estimate heart rate from the RGB buffer.
    # Needs enough samples and about 8 seconds of data.
    if len(rgb_buffer) < 60:
        return None

    rgb = np.asarray(rgb_buffer, dtype=np.float64)
    times = np.asarray(time_buffer, dtype=np.float64)

    duration = times[-1] - times[0]
    if duration < 8.0:
        return None

    # Resample to a uniform 30 Hz grid.
    target_fs = 30.0
    n = int(duration * target_fs)
    if n < 256:
        return None

    times0 = times - times[0]
    t_uniform = np.linspace(0.0, duration, n)

    r = np.interp(t_uniform, times0, rgb[:, 0])
    g = np.interp(t_uniform, times0, rgb[:, 1])
    b = np.interp(t_uniform, times0, rgb[:, 2])

    X = np.vstack([r, g, b])

    # Standardize each channel.
    for i in range(3):
        X[i] -= np.mean(X[i])
        std = np.std(X[i])
        if std > 1e-8:
            X[i] /= std

    try:
        components = fastica(X)
    except Exception:
        return None

    window = np.hamming(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / target_fs)
    bpm_values = freqs * 60.0

    # Keep the heart-rate search in a plausible range.
    valid = (bpm_values >= RPPG_MIN_BPM) & (bpm_values <= RPPG_MAX_BPM)
    if not np.any(valid):
        return None

    best_bpm = None
    best_snr = 0.0

    # Pick the ICA component with the clearest peak.
    for comp in components:
        comp = comp - np.mean(comp)
        spectrum = np.fft.rfft(comp * window)
        power = np.abs(spectrum) ** 2

        valid_power = power[valid]
        valid_bpm = bpm_values[valid]

        peak_idx = int(np.argmax(valid_power))
        peak_power = float(valid_power[peak_idx])
        noise_floor = float(np.median(valid_power))

        if noise_floor <= 0:
            continue

        snr = peak_power / noise_floor

        if snr > best_snr:
            best_snr = snr
            best_bpm = float(valid_bpm[peak_idx])

    # Require a minimum SNR.
    if best_bpm is None or best_snr < 5.0:
        return None

    return int(round(best_bpm))


def extract_face_roi_rgb(frame, landmarks, width, height):
    # Two cheek ROIs on the lower face.
    # Their position follows the detected face size.
    points = []

    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)

        if 0 <= x < width and 0 <= y < height:
            points.append([x, y])

    if not points:
        return None

    points = np.array(points)

    x_min = max(np.min(points[:, 0]), 0)
    x_max = min(np.max(points[:, 0]), width - 1)
    y_min = max(np.min(points[:, 1]), 0)
    y_max = min(np.max(points[:, 1]), height - 1)

    face_w = x_max - x_min
    face_h = y_max - y_min
    if face_w <= 0 or face_h <= 0:
        return None

    # Use a band across the lower half of the face.
    roi_y1 = int(y_min + 0.50 * face_h)
    roi_y2 = int(y_min + 0.75 * face_h)

    # Left and right cheek positions.
    left_x1 = int(x_min + 0.28 * face_w)
    left_x2 = int(x_min + 0.45 * face_w)
    right_x1 = int(x_min + 0.55 * face_w)
    right_x2 = int(x_min + 0.72 * face_w)

    # Keep the ROIs inside the frame.
    roi_y1 = max(0, roi_y1)
    roi_y2 = min(height - 1, roi_y2)
    left_x1 = max(0, left_x1)
    left_x2 = min(width - 1, left_x2)
    right_x1 = max(0, right_x1)
    right_x2 = min(width - 1, right_x2)

    left_roi = frame[roi_y1:roi_y2, left_x1:left_x2]
    right_roi = frame[roi_y1:roi_y2, right_x1:right_x2]

    if left_roi.size == 0 or right_roi.size == 0:
        return None

    if SHOW_RPPG_ROI:
        # Draw the two ROIs in debug mode.
        cv2.rectangle(frame, (left_x1, roi_y1), (left_x2, roi_y2), (255, 0, 0), 2)
        cv2.rectangle(frame, (right_x1, roi_y1), (right_x2, roi_y2), (255, 0, 0), 2)

    # Return the mean RGB value.
    mean_bgr_left = np.mean(left_roi.reshape(-1, 3), axis=0)
    mean_bgr_right = np.mean(right_roi.reshape(-1, 3), axis=0)
    mean_bgr = 0.5 * (mean_bgr_left + mean_bgr_right)

    return mean_bgr[::-1]


def draw_bottom_overlay(frame, driver_state, bpm, distraction_subtype=None):
    height, width, _ = frame.shape

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - UI_BOTTOM_BAR_HEIGHT), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Build the overlay text.
    bpm_text = "--" if bpm is None else str(bpm)
    bpm_message = f"Heart rate: {bpm_text} BPM"
    state_message = f"State: {driver_state}"

    # UI colors: focused=green, long/sleep=red, short/other=orange.
    if driver_state == "Focused on the road":
        state_color = (0, 255, 0)
    elif "long" in driver_state or driver_state == "Sleep":
        state_color = (0, 0, 255)
    else:
        state_color = (0, 165, 255)

    # Add the subtype only in debug mode.
    display_state = state_message
    if SHOW_DEBUG_TEXT and distraction_subtype is not None and "Distracted" in driver_state:
        display_state = f"{state_message} ({distraction_subtype})"

    y_text = height - 18

    cv2.putText(frame, bpm_message, (20, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, UI_BOTTOM_FONT_SCALE, (0, 255, 255), UI_BOTTOM_THICKNESS)

    text_size, _ = cv2.getTextSize(
        display_state,
        cv2.FONT_HERSHEY_SIMPLEX,
        UI_BOTTOM_FONT_SCALE,
        UI_BOTTOM_THICKNESS
    )

    cv2.putText(frame, display_state, (width - text_size[0] - 20, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, UI_BOTTOM_FONT_SCALE, state_color, UI_BOTTOM_THICKNESS)


def draw_debug_timers(
    frame,
    width,
    is_calibrated,
    eyes_closed_secs,
    owl_long_secs,
    owl_short_secs,
    lizard_long_secs,
    lizard_short_secs,
):
    # Draw the debug timer box in the top-right.
    if not SHOW_DEBUG_TEXT:
        return

    lines = []
    if not is_calibrated:
        lines.append("Calibrating...")

    lines.extend([
        f"Microsleep (>=4s): {eyes_closed_secs:.1f}s",
        f"Sleep (>=7s): {eyes_closed_secs:.1f}s",
        f"Owl long (>=5s): {owl_long_secs:.1f}s",
        f"Owl short (10s/30s): {owl_short_secs:.1f}s",
        f"Lizard long (>=5s): {lizard_long_secs:.1f}s",
        f"Lizard short (10s/30s): {lizard_short_secs:.1f}s",
    ])

    margin = 10
    line_h = 18
    top_y = 10

    # Measure the widest line first.
    max_w = 0
    for text in lines:
        (tw, th), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            UI_DEBUG_FONT_SCALE,
            UI_DEBUG_THICKNESS,
        )
        max_w = max(max_w, tw)

    box_w = max_w + 2 * margin
    box_h = len(lines) * line_h + margin
    x2 = width - margin
    x1 = max(x2 - box_w, 0)
    y1 = top_y
    y2 = top_y + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    y = y1 + margin + 12
    for text in lines:
        (tw, _), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            UI_DEBUG_FONT_SCALE,
            UI_DEBUG_THICKNESS,
        )
        x = width - margin - tw
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            UI_DEBUG_FONT_SCALE,
            (230, 230, 230),
            UI_DEBUG_THICKNESS,
        )
        y += line_h


def get_head_yaw_score(face_landmarks):
    nose = face_landmarks[NOSE_TIP]

    left_face = face_landmarks[FACE_LEFT]
    right_face = face_landmarks[FACE_RIGHT]

    left_eye_center_x = 0.5 * (
        face_landmarks[LEFT_EYE_LEFT].x + face_landmarks[LEFT_EYE_RIGHT].x
    )
    right_eye_center_x = 0.5 * (
        face_landmarks[RIGHT_EYE_LEFT].x + face_landmarks[RIGHT_EYE_RIGHT].x
    )

    eye_mid_x = 0.5 * (left_eye_center_x + right_eye_center_x)

    face_center_x = 0.5 * (left_face.x + right_face.x)
    face_width = abs(right_face.x - left_face.x)

    # Stop if the face width is invalid.
    if face_width <= 1e-6:
        return 0.0

    # Combine three lateral nose cues:
    # nose vs eyes, nose vs face center, and edge asymmetry.
    nose_vs_eyes = (nose.x - eye_mid_x) / face_width
    nose_vs_face = (nose.x - face_center_x) / face_width

    left_distance = abs(nose.x - left_face.x)
    right_distance = abs(right_face.x - nose.x)

    side_asymmetry = (right_distance - left_distance) / face_width

    # Weighted mix for a stable yaw score.
    yaw_score = (
        0.45 * nose_vs_eyes +
        0.35 * nose_vs_face +
        0.20 * side_asymmetry
    )

    return float(yaw_score)


def get_average_ear(face_landmarks, width, height):
    left_ear = eye_aspect_ratio(
        face_landmarks, width, height,
        LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
        LEFT_EYE_LEFT, LEFT_EYE_RIGHT
    )

    right_ear = eye_aspect_ratio(
        face_landmarks, width, height,
        RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
        RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT
    )

    # Average both eyes to cut noise.
    return 0.5 * (left_ear + right_ear)


def get_iris_offset(face_landmarks):
    left_eye_left = face_landmarks[LEFT_EYE_LEFT].x
    left_eye_right = face_landmarks[LEFT_EYE_RIGHT].x
    right_eye_left = face_landmarks[RIGHT_EYE_LEFT].x
    right_eye_right = face_landmarks[RIGHT_EYE_RIGHT].x

    left_eye_center = 0.5 * (left_eye_left + left_eye_right)
    right_eye_center = 0.5 * (right_eye_left + right_eye_right)

    left_eye_width = abs(left_eye_right - left_eye_left)
    right_eye_width = abs(right_eye_right - right_eye_left)

    left_iris_x = np.mean([face_landmarks[i].x for i in LEFT_IRIS])
    right_iris_x = np.mean([face_landmarks[i].x for i in RIGHT_IRIS])

    # Avoid division by zero.
    if left_eye_width <= 1e-6 or right_eye_width <= 1e-6:
        return 0.0

    left_offset = (left_iris_x - left_eye_center) / left_eye_width
    right_offset = (right_iris_x - right_eye_center) / right_eye_width

    return float(0.5 * (left_offset + right_offset))


def remove_old_window_samples(window, current_time):
    # Drop samples outside the sliding window.
    while window and current_time - window[0][0] > SHORT_DISTRACTION_WINDOW:
        window.popleft()


def window_total_time(window):
    # Sum the durations in the window.
    return sum(v for _, v in window)


# =========================
# Main
# =========================

def parse_arguments():
    global SHOW_LANDMARK_VISUALIZATION, SHOW_DEBUG_TEXT
    # Enable debug view from the command line.
    if "--debug" in sys.argv:
        SHOW_LANDMARK_VISUALIZATION = True
        SHOW_DEBUG_TEXT = True
        print("Debug mode enabled.")


def run_dms():
    # Make sure the MediaPipe task file exists first.
    download_model()

    # Open the default camera. Use DirectShow on Windows.
    if os.name == "nt":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: cannot open laptop camera.")
        return

    ret, test_frame = cap.read()
    if not ret:
        print("Error: cannot read first frame.")
        cap.release()
        return

    height, width, _ = test_frame.shape

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0

    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_NAME,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height)
    )

    # Configure FaceLandmarker for video.
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )

    # Create the landmarker once and reuse it.
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    print("DMS started. Press 'q' to quit.")
    print(f"Output video will be saved as: {OUTPUT_VIDEO_NAME}")
    print("Initial calibration: look straight at the camera with eyes open.")

    start_time = time.time()
    prev_frame_time = start_time

    # Calibration and runtime buffers.
    gaze_samples = deque()
    ear_samples = deque()
    iris_samples = deque()

    # Baselines and thresholds.
    gaze_baseline = None
    gaze_threshold_on = None
    gaze_threshold_off = None
    gaze_ema = None

    iris_baseline = None
    iris_threshold_on = None
    iris_threshold_off = None
    iris_ema = None

    ear_open_baseline = None
    ear_close_threshold = None
    ear_open_threshold = None
    ear_ema = None
    eyes_closed_state = False

    # Distraction state.
    current_distraction = None
    distraction_start_time = None

    active_short_distraction = None
    short_alarm_forward_start = None

    # Sliding windows for short distractions.
    owl_dt_window = deque()
    lizard_dt_window = deque()

    # State timers.
    focused_start_time = None
    eyes_closed_start_time = None
    eyes_open_start_time = None
    face_lost_start_time = None
    fatigue_alarm_state = None
    distraction_subtype = None

    # rPPG buffers.
    rgb_buffer = deque()
    time_buffer = deque()
    bpm = None
    last_bpm_update = 0.0

    driver_state = "Focused on the road"

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: cannot read frame.")
            break

        current_time = time.time()
        dt = max(current_time - prev_frame_time, 0.0)
        prev_frame_time = current_time

        timestamp_ms = int((current_time - start_time) * 1000)

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        is_calibrated = (
            gaze_baseline is not None and
            ear_open_baseline is not None and
            iris_baseline is not None
        )

        if result.face_landmarks:
            face_lost_start_time = None
            face_landmarks = result.face_landmarks[0]

            draw_landmark_visualization(frame, face_landmarks, width, height)

            mean_rgb = extract_face_roi_rgb(frame, face_landmarks, width, height)

            if mean_rgb is not None:
                rgb_buffer.append(mean_rgb)
                time_buffer.append(current_time)

            while time_buffer and current_time - time_buffer[0] > RPPG_WINDOW_SECONDS:
                time_buffer.popleft()
                rgb_buffer.popleft()

            if current_time - last_bpm_update >= 2.0:
                estimated_bpm = estimate_bpm_from_rgb(rgb_buffer, time_buffer)

                if estimated_bpm is not None:
                    if bpm is None:
                        bpm = estimated_bpm
                    elif abs(estimated_bpm - bpm) <= 20:
                        bpm = int(round(0.85 * bpm + 0.15 * estimated_bpm))

                last_bpm_update = current_time

            raw_gaze = get_head_yaw_score(face_landmarks)
            raw_ear = get_average_ear(face_landmarks, width, height)
            raw_iris = get_iris_offset(face_landmarks)

            if gaze_ema is None:
                gaze_ema = raw_gaze
            else:
                gaze_ema = (1.0 - GAZE_EMA_ALPHA) * gaze_ema + GAZE_EMA_ALPHA * raw_gaze

            if iris_ema is None:
                iris_ema = raw_iris
            else:
                iris_ema = (1.0 - IRIS_EMA_ALPHA) * iris_ema + IRIS_EMA_ALPHA * raw_iris

            if ear_ema is None:
                ear_ema = raw_ear
            else:
                if raw_ear < ear_ema:
                    alpha = EAR_EMA_ALPHA_CLOSE
                else:
                    alpha = EAR_EMA_ALPHA_OPEN

                ear_ema = (1.0 - alpha) * ear_ema + alpha * raw_ear

            if not is_calibrated:
                gaze_samples.append(gaze_ema)
                ear_samples.append(ear_ema)
                iris_samples.append(iris_ema)

                if len(gaze_samples) >= MIN_CALIBRATION_SAMPLES:
                    gaze_array = np.array(gaze_samples, dtype=np.float64)
                    ear_array = np.array(ear_samples, dtype=np.float64)
                    iris_array = np.array(iris_samples, dtype=np.float64)

                    temp_gaze_std = float(robust_std(gaze_array))
                    temp_iris_std = float(robust_std(iris_array))

                    if temp_gaze_std > 0.060 or temp_iris_std > 0.100:
                        print("Calibration unstable: keep head still and look forward.")
                        gaze_samples.clear()
                        ear_samples.clear()
                        iris_samples.clear()
                        driver_state = "Calibrating again"
                    else:
                        gaze_baseline = float(np.median(gaze_array))
                        iris_baseline = float(np.median(iris_array))
                        ear_open_baseline = float(np.percentile(ear_array, 70))

                        # Compute the calibration thresholds.
                        # Lower gaze thresholds catch head turns earlier.
                        gaze_threshold_on = min(max(0.048, 2.650 * temp_gaze_std), 0.112)
                        gaze_threshold_off = min(max(0.034, 1.950 * temp_gaze_std), 0.087)

                        # Iris thresholds.
                        # Slightly lower values make lizard detection easier.
                        iris_threshold_on = min(max(0.070, 3.300 * temp_iris_std), 0.140)
                        iris_threshold_off = min(max(0.052, 2.500 * temp_iris_std), 0.110)

                        # Less sensitive eye-closure detection.
                        # Close later, reopen sooner.
                        ear_close_threshold = max(0.07, ear_open_baseline * 0.40)
                        ear_open_threshold = max(0.11, ear_open_baseline * 0.62)

                        print("Calibration completed.")
                        print(f"gaze baseline={gaze_baseline:+.4f}")
                        print(f"gaze thr on={gaze_threshold_on:.4f}, off={gaze_threshold_off:.4f}")
                        print(f"iris baseline={iris_baseline:+.4f}")
                        print(f"iris thr on={iris_threshold_on:.4f}, off={iris_threshold_off:.4f}")
                        print(f"EAR open baseline={ear_open_baseline:.3f}")
                        print(f"EAR close threshold={ear_close_threshold:.3f}, open threshold={ear_open_threshold:.3f}")

                driver_state = "Focused on the road"

            else:
                if eyes_closed_state:
                    eyes_closed_state = ear_ema < ear_open_threshold
                else:
                    eyes_closed_state = ear_ema < ear_close_threshold

                raw_eyes_closed = eyes_closed_state

                gaze_diff = abs(gaze_ema - gaze_baseline)
                iris_diff = abs(iris_ema - iris_baseline)

                remove_old_window_samples(owl_dt_window, current_time)
                remove_old_window_samples(lizard_dt_window, current_time)

                if raw_eyes_closed:
                    eyes_open_start_time = None

                    if eyes_closed_start_time is None:
                        eyes_closed_start_time = current_time

                    closed_duration = current_time - eyes_closed_start_time

                    if closed_duration < BLINK_GRACE_TIME:
                        pass
                    else:
                        # Reset distraction tracking once the blink is long enough.
                        current_distraction = None
                        distraction_start_time = None
                        focused_start_time = None

                        # Set the fatigue state from the closed-eye time.
                        if closed_duration >= SLEEP_TIME:
                            fatigue_alarm_state = "Sleep"
                            driver_state = "Sleep"
                        elif closed_duration >= MICROSLEEP_TIME:
                            fatigue_alarm_state = "Microsleep"
                            driver_state = "Microsleep"
                        else:
                            driver_state = "Focused on the road"

                else:
                    if eyes_closed_start_time is not None and (current_time - eyes_closed_start_time) < BLINK_GRACE_TIME:
                        # Short blink: do not count it as fatigue.
                        eyes_closed_start_time = None
                        fatigue_alarm_state = None

                    if eyes_open_start_time is None:
                        eyes_open_start_time = current_time

                    eyes_open_duration = current_time - eyes_open_start_time

                    if eyes_open_duration >= FOCUSED_RESET_TIME:
                        # Eyes have been open long enough to reset blink timers.
                        eyes_closed_start_time = None

                        # Clear the fatigue alarm once the eyes stay open.
                        if fatigue_alarm_state is not None:
                            fatigue_alarm_state = None

                    if fatigue_alarm_state is None:
                        # Reset blink timers on reopen when no fatigue alarm is active.
                        eyes_closed_start_time = None

                        # Use separate thresholds for OWL entry and exit.
                        if current_distraction == "owl":
                            owl_detected = gaze_diff > gaze_threshold_off
                        else:
                            owl_detected = gaze_diff > gaze_threshold_on

                        # Only classify lizard when the head is really forward.
                        lizard_head_forward_threshold = gaze_threshold_on / LIZARD_HEAD_FORWARD_MULT
                        head_forward_for_lizard = gaze_diff < lizard_head_forward_threshold

                        if current_distraction == "lizard":
                            lizard_detected = (
                                (not owl_detected) and
                                head_forward_for_lizard and
                                (iris_diff > iris_threshold_off)
                            )
                        else:
                            lizard_detected = (
                                (not owl_detected) and
                                head_forward_for_lizard and
                                (iris_diff > iris_threshold_on)
                            )

                        if owl_detected:
                            new_distraction = "owl"
                        elif lizard_detected:
                            new_distraction = "lizard"
                        else:
                            new_distraction = None

                        if new_distraction != current_distraction:
                            current_distraction = new_distraction
                            distraction_start_time = current_time if new_distraction is not None else None

                        if current_distraction == "lizard":
                            # Lizard is still active, so do not start the forward timer.
                            focused_start_time = None
                            short_alarm_forward_start = None

                            lizard_duration = current_time - distraction_start_time

                            # Keep adding lizard time to the 30-second buffer.
                            lizard_dt_window.append((current_time, dt))
                            lizard_cumulative = window_total_time(lizard_dt_window)

                            if lizard_duration >= LONG_DISTRACTION_TIME:
                                driver_state = "Distracted (long)"
                                active_short_distraction = None
                            elif lizard_cumulative >= SHORT_DISTRACTION_TOTAL:
                                driver_state = "Distracted (short)"
                                active_short_distraction = "lizard"
                            elif active_short_distraction == "lizard":
                                driver_state = "Distracted (short)"
                            else:
                                driver_state = "Focused on the road"

                        elif current_distraction == "owl":
                            # Owl is still active, so do not start the forward timer.
                            focused_start_time = None
                            short_alarm_forward_start = None

                            owl_duration = current_time - distraction_start_time

                            # Keep adding owl time to the 30-second buffer.
                            owl_dt_window.append((current_time, dt))
                            owl_cumulative = window_total_time(owl_dt_window)

                            if owl_duration >= LONG_DISTRACTION_TIME:
                                driver_state = "Distracted (long)"
                                active_short_distraction = None
                            elif owl_cumulative >= SHORT_DISTRACTION_TOTAL:
                                driver_state = "Distracted (short)"
                                active_short_distraction = "owl"
                            elif active_short_distraction == "owl":
                                driver_state = "Distracted (short)"
                            else:
                                driver_state = "Focused on the road"

                        else:
                            # No distraction active: the driver is looking forward.
                            if focused_start_time is None:
                                focused_start_time = current_time

                            # Keep the short alarm on until the driver stays forward long enough.
                            if active_short_distraction in ("owl", "lizard"):
                                if short_alarm_forward_start is None:
                                    short_alarm_forward_start = current_time

                                forward_duration = current_time - short_alarm_forward_start

                                if forward_duration < FOCUSED_RESET_TIME:
                                    driver_state = "Distracted (short)"
                                else:
                                    active_short_distraction = None
                                    short_alarm_forward_start = None
                                    driver_state = "Focused on the road"
                            else:
                                short_alarm_forward_start = None
                                driver_state = "Focused on the road"

                            if current_time - focused_start_time >= FOCUSED_RESET_TIME:
                                gaze_baseline = 0.995 * gaze_baseline + 0.005 * gaze_ema
                                iris_baseline = 0.995 * iris_baseline + 0.005 * iris_ema
                                ear_open_baseline = 0.995 * ear_open_baseline + 0.005 * ear_ema

                                ear_close_threshold = max(0.07, ear_open_baseline * 0.40)
                                ear_open_threshold = max(0.11, ear_open_baseline * 0.62)

                    else:
                        # Keep the active alarm until the eyes stay open long enough.
                        driver_state = fatigue_alarm_state

            if SHOW_DEBUG_TEXT:
                # Debug timers for the assignment.
                if is_calibrated and eyes_closed_start_time is not None and eyes_closed_state:
                    eyes_closed_secs = max(0.0, current_time - eyes_closed_start_time)
                else:
                    eyes_closed_secs = 0.0

                owl_long_secs = (
                    max(0.0, current_time - distraction_start_time)
                    if (is_calibrated and current_distraction == "owl" and distraction_start_time is not None)
                    else 0.0
                )

                lizard_long_secs = (
                    max(0.0, current_time - distraction_start_time)
                    if (is_calibrated and current_distraction == "lizard" and distraction_start_time is not None)
                    else 0.0
                )

                owl_short_secs = float(window_total_time(owl_dt_window)) if is_calibrated else 0.0
                lizard_short_secs = float(window_total_time(lizard_dt_window)) if is_calibrated else 0.0

                draw_debug_timers(
                    frame,
                    width,
                    is_calibrated,
                    eyes_closed_secs,
                    owl_long_secs,
                    owl_short_secs,
                    lizard_long_secs,
                    lizard_short_secs,
                )

        else:
            if face_lost_start_time is None:
                face_lost_start_time = current_time

            face_lost_duration = current_time - face_lost_start_time

            if NO_FACE_AS_DISTRACTION and is_calibrated and face_lost_duration >= NO_FACE_GRACE_TIME:
                eyes_closed_start_time = None
                eyes_open_start_time = None
                focused_start_time = None
                fatigue_alarm_state = None

                if current_distraction != "owl":
                    current_distraction = "owl"
                    distraction_start_time = current_time

                short_alarm_forward_start = None

                owl_duration = current_time - distraction_start_time

                # Keep adding no-face time to the owl buffer.
                owl_dt_window.append((current_time, dt))
                remove_old_window_samples(owl_dt_window, current_time)

                owl_cumulative = window_total_time(owl_dt_window)

                if owl_duration >= LONG_DISTRACTION_TIME:
                    driver_state = "Distracted (long)"
                elif owl_cumulative >= SHORT_DISTRACTION_TOTAL:
                    driver_state = "Distracted (short)"
                else:
                    driver_state = "Focused on the road"
            else:
                current_distraction = None
                distraction_start_time = None
                active_short_distraction = None
                short_alarm_forward_start = None
                driver_state = "Face not detected"

        display_distraction_subtype = current_distraction if current_distraction is not None else active_short_distraction

        draw_bottom_overlay(frame, driver_state, bpm, display_distraction_subtype)

        video_writer.write(frame)

        cv2.imshow("Driver Monitoring System", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    face_landmarker.close()
    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

    print(f"Video saved as: {OUTPUT_VIDEO_NAME}")


if __name__ == "__main__":
    parse_arguments()
    run_dms()