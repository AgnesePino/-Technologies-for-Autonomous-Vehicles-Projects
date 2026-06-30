#!/usr/bin/env python3
"""GOLD lane detection on PandaSet images and YOLO obstacle detection
"""

import argparse
import glob
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# -------------------------------------------------
# 1) Camera intrinsics + BEV setup
# -------------------------------------------------

IMG_W: int   = 1920
IMG_H: int   = 1080
FX:    float = 1970.0
FY:    float = 1970.0
CX:    float = 970.0
CY:    float = 483.0

CAMERA_HEIGHT_M: float = 1.660
PITCH_DEG:       float = 0.0

# BEV world range (meters)
X_MIN_M: float = -4.8
X_MAX_M: float =  4.8
Y_MIN_M: float =  5.5
Y_MAX_M: float = 35.0

# Fixed BEV canvas (pixels)
BEV_W: int = 1000
BEV_H: int = 1200

# Rough resolution (px/m) from BEV width and lateral world span
PIXELS_PER_METER: int = int(round(BEV_W / max((X_MAX_M - X_MIN_M), 1e-6)))

# Quick correction factor (flat ground math tends to overestimate distance)
RANGE_CORRECTION: float = 0.85

# Extra scale applied ONLY to YOLO obstacle distances (after the pinhole model).
# If obstacle distances look too big, keep this < 1.0.
OBSTACLE_DISTANCE_SCALE: float = 0.80


# -------------------------------------------------
# 2) Lane detection
# -------------------------------------------------

# GOLD filter half-width
GOLD_M: int = 4

# Lane geometry (meters → BEV pixels)
LANE_WIDTH_M_MIN:     float = 2.6
LANE_WIDTH_M_MAX:     float = 4.4
LANE_WIDTH_M_NOMINAL: float = 3.5

# Sliding-window parameters
SW_N_WINDOWS: int   = 14
SW_MARGIN:    int   = 60
SW_MINPIX:    int   = 8

# Quality thresholds for boundary acceptance
MIN_ACTIVE_WINDOWS_RATIO: float = 0.40
MIN_INLIER_RATIO:         float = 0.50
MIN_SPAN_RATIO:           float = 0.55
MAX_SIGMA_PX:             float = 20.0
MAX_SLOPE_ABS:            float = 0.32
MIN_MASK_FILL_RATIO:      float = 0.12

# Single-lane acceptance (stricter)
SINGLE_MIN_ACTIVE_WINDOWS_RATIO: float = 0.48
SINGLE_MIN_INLIER_RATIO:         float = 0.80
SINGLE_MIN_SPAN_RATIO:           float = 0.65
SINGLE_MAX_SIGMA_PX:             float = 13.0
SINGLE_MAX_SLOPE_ABS:            float = 0.13
SINGLE_MAX_P90_PX:               float = 20.0

# Pair geometry gates
PAIR_WIDTH_NEAR_LO: float = 0.18   # fractions of BEV_W
PAIR_WIDTH_NEAR_HI: float = 0.58
PAIR_WIDTH_FAR_LO:  float = 0.08
PAIR_WIDTH_FAR_HI:  float = 0.72
PAIR_WIDTH_FAR_NEAR_LO: float = 0.72
PAIR_WIDTH_FAR_NEAR_HI: float = 2.20
PAIR_MIN_SPAN:      float = 0.55
PAIR_LEFT_MIN_SPAN: float = 0.60

# Histogram seeding
SEED_THRESHOLD: float = 24.0
SEED_MAX_PEAKS: int   = 4
SEED_DEAD_ZONE: int   = 45

# Lane type classification
CLASSIFY_TYPE: bool = True

# Intersection / clutter rejection
INTERSECTION_CHECK_ROW_START: float = 0.60
INTERSECTION_CHECK_ROW_END:   float = 0.95
MAX_HORIZ_ENERGY_RATIO:       float = 0.16
MAX_CENTER_OCC_RATIO:         float = 0.12

# Temporal smoothing
MAX_TRACK_LOST_FRAMES: int   = 6
LANE_TRACK_ALPHA:      float = 0.80
LANE_MAX_JUMP_PX:      int   = 50

# Temporal stabilization for lane type (continuous/dashed)
TYPE_CONFIRM_FRAMES:   int   = 3

# YOLO obstacle classes
YOLO_TRACKED_CLASSES = {
    "person", "bicycle", "motorcycle", "car", "bus", "truck", "train",
}


# -------------------------------------------------
# 3) Small data containers
# -------------------------------------------------

@dataclass
class BoundaryFit:
    """Stores one fitted lane boundary in BEV pixel space (x = slope*y + offset)."""
    side:     str           # "left" | "right"
    slope:    float
    offset:   float
    x_near:   float         # x at bottom of BEV (near field)
    x_far:    float         # x at top  of BEV (far field)
    votes:    float
    style:    str           # "continuous" | "dashed" | "inferred"
    quality:  Dict[str, float] = field(default_factory=dict)


@dataclass
class TrackState:
    left_col:      Optional[int]   = None
    right_col:     Optional[int]   = None
    lane_width_px: Optional[float] = None
    center_col:    Optional[float] = None
    lost_count:    int             = 0

    # Stable lane type labels (to avoid flicker)
    left_type:     str             = "unknown"
    right_type:    str             = "unknown"
    left_pending_type:  Optional[str] = None
    right_pending_type: Optional[str] = None
    left_pending_count:  int          = 0
    right_pending_count: int          = 0


@dataclass
class Obstacle:
    label:      str
    confidence: float
    bbox:       Tuple[int, int, int, int]
    distance_m: Optional[float]
    in_lane:    bool


# Global tracker state (kept across frames)
TRACK = TrackState()

# Cached projection matrix (computed lazily)
_P_WORLD_TO_IMG: Optional[np.ndarray] = None


# -------------------------------------------------
# 4) Camera geometry helpers
# -------------------------------------------------

def _build_projection_matrix() -> np.ndarray:
    """Build the 3x4 projection matrix I use for ground-plane projection."""
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
    theta = np.deg2rad(PITCH_DEG)
    R_base = np.array(
        [[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64
    )
    R_pitch = np.array(
        [[1, 0, 0], [0, math.cos(theta), -math.sin(theta)],
         [0, math.sin(theta),  math.cos(theta)]], dtype=np.float64
    )
    R = R_pitch @ R_base
    C = np.array([0.0, 0.0, CAMERA_HEIGHT_M], dtype=np.float64)
    t = -R @ C
    return K @ np.column_stack([R, t])


def _get_proj() -> np.ndarray:
    global _P_WORLD_TO_IMG
    if _P_WORLD_TO_IMG is None:
        _P_WORLD_TO_IMG = _build_projection_matrix()
    return _P_WORLD_TO_IMG


def world_to_image(xw: float, yw: float) -> Optional[Tuple[int, int]]:
    """Project a ground-plane point (x right, y forward) to a camera pixel."""
    if yw <= 1e-6:
        return None
    p = _get_proj() @ np.array([float(yw), -float(xw), 0.0, 1.0], dtype=np.float64)
    if p[2] <= 1e-6:
        return None
    u, v = p[0] / p[2], p[1] / p[2]
    return int(round(u)), int(round(v))


def image_horizon() -> float:
    return CY + FY * math.tan(math.radians(PITCH_DEG))


def image_row_to_distance(v: float) -> Optional[float]:
    denom = v - image_horizon()
    if denom <= 1.0:
        return None
    return (FY * CAMERA_HEIGHT_M / denom) * RANGE_CORRECTION


def image_u_to_world_x(u: float, z: float) -> float:
    return (u - CX) * z / FX


# -------------------------------------------------
# 5) BEV / IPM setup
# -------------------------------------------------

def build_ipm_maps() -> Tuple[np.ndarray, np.ndarray]:
    """Build OpenCV remap tables for the BEV warp."""
    cols = np.arange(BEV_W, dtype=np.float32)
    rows = np.arange(BEV_H, dtype=np.float32)
    Xw = X_MIN_M + cols * (X_MAX_M - X_MIN_M) / BEV_W
    Yw = Y_MAX_M - rows * (Y_MAX_M - Y_MIN_M) / BEV_H
    Xg, Yg = np.meshgrid(Xw, Yw)

    if abs(PITCH_DEG) < 1e-6:
        map_x = FX * Xg / np.maximum(Yg, 1e-6) + CX
        map_y = FY * CAMERA_HEIGHT_M / np.maximum(Yg, 1e-6) + CY
    else:
        pitch = math.radians(PITCH_DEG)
        Xc = Xg
        Yc = Yg * math.cos(pitch) + CAMERA_HEIGHT_M * math.sin(pitch)
        Zc = Yg * math.sin(pitch) + CAMERA_HEIGHT_M * math.cos(pitch)
        map_x = FX * Xc / np.maximum(Zc, 1e-6) + CX
        map_y = FY * Yc / np.maximum(Zc, 1e-6) + CY

    valid = (
        (Yg > 0.1)
        & (map_x >= 0) & (map_x < IMG_W - 1)
        & (map_y >= 0) & (map_y < IMG_H - 1)
    )
    map_x[~valid] = -1
    map_y[~valid] = -1
    return map_x.astype(np.float32), map_y.astype(np.float32)


def compute_homography_bev_to_img() -> np.ndarray:
    """Homography that maps BEV pixels to the image (ground plane only)."""
    P = _get_proj()
    H_gnd = P[:, [0, 1, 3]]
    dx = X_MAX_M - X_MIN_M
    dy = Y_MAX_M - Y_MIN_M
    T = np.array([
        [0.0,       -dy / BEV_H, Y_MAX_M ],
        [-dx / BEV_W, 0.0,      -X_MIN_M ],
        [0.0,        0.0,        1.0     ],
    ], dtype=np.float64)
    return H_gnd @ T


def warp_to_bev(
    img: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    H_bev_to_img: Optional[np.ndarray] = None,
) -> np.ndarray:
    if H_bev_to_img is not None:
        return cv2.warpPerspective(
            img, H_bev_to_img, (BEV_W, BEV_H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def build_valid_mask(map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Mask of BEV pixels that map to a valid image location."""
    ok = (map_x >= 0) & (map_x < IMG_W - 1) & (map_y >= 0) & (map_y < IMG_H - 1)
    return (ok.astype(np.uint8) * 255)


# BEV coordinate helpers (pixel <-> meters)
def col_to_world_x(col: float) -> float:
    return X_MIN_M + float(col) * (X_MAX_M - X_MIN_M) / BEV_W


def world_x_to_bev_col(x: float) -> int:
    return int(np.clip(round((x - X_MIN_M) * BEV_W / (X_MAX_M - X_MIN_M)), 0, BEV_W - 1))


def world_y_to_bev_row(y: float) -> int:
    return int(np.clip(round((Y_MAX_M - y) * BEV_H / (Y_MAX_M - Y_MIN_M)), 0, BEV_H - 1))


# -------------------------------------------------
# 6) BEV image cleanup (grayscale)
# -------------------------------------------------

def _percentile_stretch(gray: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Quick contrast stretch using the 2–98 percentiles (valid BEV pixels only)."""
    out  = np.zeros_like(gray)
    keep = valid > 0
    if keep.sum() < 200:
        return gray.copy()
    vals = gray[keep].astype(np.float32)
    lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
    if hi - lo < 1.0:
        return gray.copy()
    scaled = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    out[keep] = scaled[keep]
    return out


def normalise_bev(bev_bgr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Turn the BEV image into a nicer grayscale for feature extraction.

    Steps: BGR->gray -> stretch -> CLAHE -> blur -> small gamma lift.
    """
    gray = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2GRAY)
    gray = _percentile_stretch(gray, valid)
    gray = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Small gamma lift: helps faint markings without blowing highlights
    tone = np.power(gray.astype(np.float32) / 255.0, 0.75)
    return np.clip(tone * 255.0, 0, 255).astype(np.uint8)


# -------------------------------------------------
# 7) Lane feature extraction (GOLD + morpho)
# -------------------------------------------------

def gold_filter(gray: np.ndarray, m: int = GOLD_M) -> np.ndarray:
    """GOLD response (Gradient Operator for Lane Detection).

    Formula: R(x) = max(0, (I(x) - I(x-m)) + (I(x) - I(x+m)))
    """
    I      = gray.astype(np.float32)
    d_left = I - np.roll(I, m,  axis=1)
    d_rght = I - np.roll(I, -m, axis=1)
    R      = np.where((d_left > 0) & (d_rght > 0), d_left + d_rght, 0.0)
    # Clean up borders and invalid pixels
    R[:, :m]  = 0.0
    R[:, -m:] = 0.0
    R[gray == 0] = 0.0
    # Also ignore the extreme left/right margins (usually noisy)
    margin = int(0.12 * R.shape[1])
    R[:, :margin]  = 0.0
    R[:, -margin:] = 0.0
    return R


def build_lane_feature_map(gray: np.ndarray) -> np.ndarray:
    """Build a lane "response" map by mixing a few simple cues.

    I combine:
    - GOLD filter
    - top-hat at two different sizes (thin bright stripes)
    - lateral Sobel gradient (edges)

    It ends up being more stable than using just one of them.
    """
    gold = gold_filter(gray)
    gold_u8 = np.clip(gold, 0, 255).astype(np.uint8)

    # Background estimate / subtraction
    bg   = cv2.morphologyEx(gray, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (61, 5)))
    flat = cv2.subtract(gray, bg)

    # Top-hat at two scales
    th_big = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (35, 7)))
    th_sml = cv2.morphologyEx(flat, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5)))

    # Lateral gradient
    gx = cv2.Sobel(flat, cv2.CV_32F, 1, 0, ksize=3)
    gx = np.abs(gx)
    if gx.max() > 0:
        gx = (255.0 * gx / gx.max()).astype(np.uint8)
    else:
        gx = np.zeros_like(gray)

    # Mix cues (weights are hand-tuned)
    morpho = cv2.addWeighted(th_big, 0.40, th_sml, 0.35, 0.0)
    morpho = cv2.addWeighted(morpho, 0.80, gx,     0.20, 0.0)

    # Final blend: GOLD + morpho/gradient
    fused = cv2.addWeighted(gold_u8, 0.45, morpho, 0.55, 0.0)
    return cv2.GaussianBlur(fused, (3, 3), 0)


# -------------------------------------------------
# 8) Binary segmentation
# -------------------------------------------------

def _binarise(feat: np.ndarray) -> np.ndarray:
    """Binarize by OR-ing global Otsu with local adaptive threshold."""
    _, otsu = cv2.threshold(feat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        feat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -5
    )
    return cv2.bitwise_or(otsu, adaptive)


def _keep_vertical_stripes(binary: np.ndarray) -> np.ndarray:
    """Morphology to kill horizontal clutter but keep vertical-ish lane stripes."""
    opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (3, 20)))
    closed  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (5, 45)))
    refined = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15)))
    return refined


def _drop_wide_blobs(mask: np.ndarray) -> np.ndarray:
    """Remove components that look too wide/squat (texture, shadows, crosswalks).
    """
    out = np.zeros_like(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Rows with tons of white pixels are usually clutter, not lanes
    dense_rows = np.mean(mask > 0, axis=1) > 0.30

    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        if area < 40 or bh < 25:
            continue
        if bh / max(bw, 1) < 1.0:
            continue
        if bw > 80 and bh < 60:
            continue
        rows_hit = np.where(np.any(labels == k, axis=1))[0]
        if len(rows_hit) > 0 and np.mean(dense_rows[rows_hit]) > 0.60:
            continue
        out[labels == k] = 255

    return out


def _clip_to_ego_corridor(mask: np.ndarray) -> np.ndarray:
    """Keep only the central BEV corridor (where the ego lane should be)."""
    h, w = mask.shape
    roi  = np.zeros_like(mask)
    cv2.rectangle(roi, (int(0.18 * w), int(0.05 * h)), (int(0.82 * w), h), 255, -1)
    return cv2.bitwise_and(mask, roi)


def segment_lane_pixels(
    bev_bgr: np.ndarray, valid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the full preprocessing on the BEV.

    Output is (gray_norm, feature_map, lane_mask).
    """
    norm     = normalise_bev(bev_bgr, valid)
    features = build_lane_feature_map(norm)
    binary   = _binarise(features)
    cleaned  = _keep_vertical_stripes(binary)
    cleaned  = _drop_wide_blobs(cleaned)
    cleaned  = _clip_to_ego_corridor(cleaned)
    cleaned  = cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    )
    return norm, features, cleaned


# -------------------------------------------------
# 9) Histogram seeds
# -------------------------------------------------

def _column_histogram(mask: np.ndarray) -> np.ndarray:
    """Smoothed column histogram in a mid-field BEV band.

    Mid-field is usually the sweet spot: close enough to see markings,
    but not too distorted.
    """
    h   = mask.shape[0]
    band = mask[int(0.35 * h): int(0.85 * h), :]
    raw  = np.sum(band > 0, axis=0).astype(np.float32)
    return cv2.GaussianBlur(raw.reshape(1, -1), (0, 0), sigmaX=12).reshape(-1)


def _find_peaks(
    hist: np.ndarray,
    lo: int, hi: int,
    threshold: float,
    n_max:    int = SEED_MAX_PEAKS,
    dead_zone: int = SEED_DEAD_ZONE,
) -> List[int]:
    """Extract up to n_max peaks from hist[lo:hi] with a dead-zone suppression."""
    lo, hi  = max(0, lo), min(len(hist), hi)
    work    = hist.copy()
    peaks: List[int] = []
    for _ in range(n_max):
        sub = work[lo:hi]
        idx = int(np.argmax(sub))
        if sub[idx] < threshold:
            break
        p = lo + idx
        peaks.append(p)
        work[max(0, p - dead_zone): min(len(work), p + dead_zone + 1)] = 0.0
    return peaks


def find_seed_columns(
    mask: np.ndarray,
) -> Tuple[List[int], List[int], np.ndarray]:
    """Return (left_seeds, right_seeds, histogram)."""
    _, w = mask.shape
    hist = _column_histogram(mask)
    # Slightly adaptive threshold: if the histogram is weak (low contrast),
    thr  = float(min(SEED_THRESHOLD, 0.55 * float(hist.max()))) if hist.size else float(SEED_THRESHOLD)
    thr  = max(10.0, thr)
    c    = w // 2
    # I avoid the extreme margins because they give me lots of false peaks.
    lefts  = _find_peaks(hist, int(0.18 * w), c - 15, thr)
    rights = _find_peaks(hist, c + 15, int(0.82 * w), thr)
    return lefts, rights, hist


# -------------------------------------------------
# 10) Sliding windows + line fit
# -------------------------------------------------

def _classify_style(mask: np.ndarray, slope: float, offset: float) -> str:
    """Quick solid vs dashed check by sampling the mask along the fitted line."""
    h, w   = mask.shape
    ys     = np.arange(int(0.20 * h), h, dtype=np.int32)
    occ    = []
    for y in ys:
        xc  = int(round(slope * y + offset))
        x0  = max(0, xc - 6)
        x1  = min(w, xc + 7)
        occ.append(1 if np.mean(mask[y, x0:x1] > 0) > 0.12 else 0)

    occ = np.array(occ, dtype=np.uint8)
    if not len(occ):
        return "dashed"

    cleaned, start = occ.copy(), None
    for i in range(len(occ)):
        if occ[i] and start is None:
            start = i
        if (not occ[i] or i == len(occ) - 1) and start is not None:
            end = i if not occ[i] else i + 1
            if end - start < 5:
                cleaned[start:end] = 0
            start = None

    fill    = float(cleaned.mean())
    switches = int(np.sum(np.abs(np.diff(cleaned)) > 0))
    if fill > 0.56 and switches < 20:
        return "continuous"
    return "dashed"


def fit_boundary(
    mask:      np.ndarray,
    seed:      int,
    side:      str,
    n_wins:    int = SW_N_WINDOWS,
    margin:    int = SW_MARGIN,
    minpix:    int = SW_MINPIX,
) -> Optional[BoundaryFit]:
    """Fit one lane side starting from a histogram seed.

    I do the usual sliding-windows thing from the bottom up. If it looks too messy
    (not enough pixels / unstable fit), I just return None.
    """
    H, W   = mask.shape
    ny, nx = mask.nonzero()
    if len(nx) == 0:
        return None

    step    = max(H // n_wins, 1)
    xcur    = seed
    buckets: List[np.ndarray] = []

    for win in range(n_wins):
        y0 = H - (win + 1) * step
        y1 = H - win * step
        x0 = max(0, xcur - margin)
        x1 = min(W, xcur + margin)

        # Hard left/right limits so the two sides don't bleed into each other.
        if side == "left":
            x1 = min(x1, int(0.62 * W))
        else:
            x0 = max(x0, int(0.38 * W))

        picked = ((ny >= y0) & (ny < y1) & (nx >= x0) & (nx < x1)).nonzero()[0]
        buckets.append(picked)
        if len(picked) > minpix:
            xcur = int(nx[picked].mean())

    n_active = sum(1 for b in buckets if len(b) > minpix)
    min_active = max(5, math.ceil(0.42 * n_wins))
    if n_active < min_active:
        return None

    idx_all = np.concatenate(buckets)
    if len(idx_all) < 40:
        return None

    px = nx[idx_all].astype(np.float32)
    py = ny[idx_all].astype(np.float32)

    # Pass 1: quick rough fit
    sl, ic = np.polyfit(py, px, 1)
    resid  = px - (sl * py + ic)
    keep   = np.abs(resid) < 32.0
    if keep.mean() < MIN_INLIER_RATIO:
        return None

    px, py = px[keep], py[keep]
    if len(px) < 30:
        return None

    # Pass 2: refit using only inliers
    sl, ic = map(float, np.polyfit(py, px, 1))
    resid  = px - (sl * py + ic)
    sigma  = float(resid.std())
    p90    = float(np.percentile(np.abs(resid), 90))
    span   = float((py.max() - py.min()) / max(H, 1))
    inlier_ratio = float(keep.mean())

    if span  < MIN_SPAN_RATIO:   return None
    if sigma > MAX_SIGMA_PX:     return None
    if abs(sl) > MAX_SLOPE_ABS:  return None

    y_near = H - 1
    y_far  = int(0.18 * H)
    x_near = sl * y_near + ic
    x_far  = sl * y_far  + ic

    # Just a few checks so left stays left, right stays right.
    cx = W / 2.0
    if side == "left"  and x_near >= cx:         return None
    if side == "right" and x_near <= cx:         return None
    if side == "left"  and not (0.16 * W <= x_near <= 0.50 * W): return None
    if side == "right" and not (0.50 * W <= x_near <= 0.82 * W): return None
    if not (-120 <= x_far <= W + 120):           return None
    if not (0 <= x_near < W):                   return None

    style = _classify_style(mask, sl, ic) if CLASSIFY_TYPE else "unknown"

    return BoundaryFit(
        side=side, slope=sl, offset=ic,
        x_near=x_near, x_far=x_far,
        votes=float(len(px)), style=style,
        quality={
            "active_wins":   float(n_active),
            "inlier_ratio":  inlier_ratio,
            "sigma":         sigma,
            "p90":           p90,
            "span":          span,
            "slope_abs":     abs(sl),
        },
    )


def _check_mask_fill(mask: np.ndarray, bf: BoundaryFit) -> bool:
    """Check the fitted line actually hits the mask often enough."""
    H, W = mask.shape
    ys   = np.arange(int(0.10 * H), H, 4, dtype=np.int32)
    half = 10
    hits = total = 0
    for y in ys:
        x  = int(round(bf.slope * y + bf.offset))
        x0 = max(0, x - half)
        x1 = min(W, x + half + 1)
        if x1 <= x0:
            continue
        total += 1
        if np.any(mask[y, x0:x1] > 0):
            hits += 1
    if total == 0:
        return False
    bf.quality["fill"] = hits / total
    return (hits / total) >= MIN_MASK_FILL_RATIO


# -------------------------------------------------
# 11) Picking the ego lane (pair/single)
# -------------------------------------------------

def _q(bf: BoundaryFit, key: str, default: float = 0.0) -> float:
    return float(bf.quality.get(key, default))


def _accept_pair(L: BoundaryFit, R: BoundaryFit, W: int) -> bool:
    """Geometry + quality gates for a left/right boundary pair."""
    if L.x_near >= R.x_near:
        return False

    w_near = R.x_near - L.x_near
    w_far  = R.x_far  - L.x_far

    if not (PAIR_WIDTH_NEAR_LO * W <= w_near <= PAIR_WIDTH_NEAR_HI * W):
        return False
    if not (PAIR_WIDTH_FAR_LO  * W <= w_far  <= PAIR_WIDTH_FAR_HI  * W):
        return False
    if R.x_far > 0.82 * W:
        return False
    if w_far < PAIR_WIDTH_FAR_NEAR_LO * w_near:
        return False
    if w_far > PAIR_WIDTH_FAR_NEAR_HI * w_near:
        return False

    lspan, rspan = _q(L, "span"), _q(R, "span")
    if lspan < PAIR_LEFT_MIN_SPAN:
        return False
    if min(lspan, rspan) < PAIR_MIN_SPAN:
        return False

    li, ri = _q(L, "inlier_ratio"), _q(R, "inlier_ratio")
    lp, rp = _q(L, "p90", 99.0),    _q(R, "p90", 99.0)
    ls, rs = _q(L, "sigma", 99.0),   _q(R, "sigma", 99.0)

    if li < 0.60 and ri < 0.60 and (lp + rp) / 2 > 28.0:
        return False
    if max(ls, rs) > 20.0 and min(li, ri) < 0.65:
        return False

    return True


def _pair_cost(L: BoundaryFit, R: BoundaryFit, W: int) -> float:
    """Pair scoring (lower is better)."""
    width  = R.x_near - L.x_near
    centre = 0.5 * (L.x_near + R.x_near)
    nom    = LANE_WIDTH_M_NOMINAL * PIXELS_PER_METER
    w_err  = abs(width  - nom)          / max(LANE_WIDTH_M_MIN * PIXELS_PER_METER, 1.0)
    c_err  = abs(centre - 0.50 * W)     / max(0.22 * W, 1.0)
    pen_r  = max(0.0, (R.x_near - 0.80 * W) / max(0.06 * W, 1.0))
    pen_l  = max(0.0, (0.15 * W - L.x_near) / max(0.06 * W, 1.0))
    bonus  = 0.08 * (_q(L, "inlier_ratio") + _q(R, "inlier_ratio"))
    return 1.60 * w_err + 0.90 * c_err + 0.70 * pen_r + 0.40 * pen_l - bonus


def _accept_single(bf: BoundaryFit, W: int) -> bool:
    """Stricter gates for a single boundary (no partner to cross-check)."""
    if bf.side == "left"  and bf.x_near < 0.20 * W: return False
    if bf.side == "right" and bf.x_near > 0.74 * W: return False

    min_active_wins = max(
        7.0, float(SINGLE_MIN_ACTIVE_WINDOWS_RATIO) * float(SW_N_WINDOWS)
    )
    return (
        _q(bf, "active_wins")  >= min_active_wins and
        _q(bf, "inlier_ratio") >= SINGLE_MIN_INLIER_RATIO and
        _q(bf, "span")         >= SINGLE_MIN_SPAN_RATIO and
        _q(bf, "sigma")        <= SINGLE_MAX_SIGMA_PX and
        _q(bf, "p90", 99.0)    <= SINGLE_MAX_P90_PX and
        _q(bf, "slope_abs")    <= SINGLE_MAX_SLOPE_ABS
    )


def _infer_partner(bf: BoundaryFit, W: int) -> BoundaryFit:
    """Mirror a detected boundary to synthesize the missing partner."""
    nom_px = int(round(LANE_WIDTH_M_NOMINAL * PIXELS_PER_METER))
    sign   = +1 if bf.side == "left" else -1
    opp    = "right" if bf.side == "left" else "left"
    return BoundaryFit(
        side=opp, slope=bf.slope, offset=bf.offset + sign * nom_px,
        x_near=bf.x_near + sign * nom_px,
        x_far =bf.x_far  + sign * nom_px,
        votes=bf.votes * 0.25, style="inferred",
    )


def pick_ego_boundaries(
    candidates: List[BoundaryFit], mask_shape: Tuple[int, int]
) -> List[BoundaryFit]:
    if not candidates:
        return []

    H, W   = mask_shape
    lefts  = [c for c in candidates if c.side == "left"]
    rights = [c for c in candidates if c.side == "right"]

    # Try all left x right combinations
    valid_pairs = [
        (_pair_cost(L, R, W), L, R)
        for L in lefts for R in rights
        if _accept_pair(L, R, W)
    ]
    if valid_pairs:
        _, bL, bR = min(valid_pairs, key=lambda t: t[0])
        return sorted([bL, bR], key=lambda b: b.x_near)

    # If no pair works, keep only the best single boundary
    singles = [c for c in candidates if _accept_single(c, W)]
    if singles:
        best = max(singles, key=lambda b: _q(b, "inlier_ratio") + _q(b, "span"))
        return [best]

    return []


# -------------------------------------------------
# 12) Clutter / intersection guard
# -------------------------------------------------

def _horizontal_energy_ratio(binary: np.ndarray, r0: int, r1: int) -> float:
    roi = (binary[r0:r1] > 0).astype(np.uint8)
    if roi.sum() == 0:
        return 0.0
    hori = cv2.morphologyEx(roi * 255, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    return int((hori > 0).sum()) / float(roi.sum())


def _center_occupancy(binary: np.ndarray, r0: int, r1: int) -> float:
    _, w = binary.shape
    c1, c2 = int(0.30 * w), int(0.70 * w)
    roi = (binary[r0:r1, c1:c2] > 0)
    return float(roi.mean()) if roi.size else 0.0


def is_heavy_clutter(binary: np.ndarray) -> bool:
    """Rough "this looks like an intersection" detector.

    It's meant to catch crosswalk-ish patterns / strong horizontal clutter.
    I check two things and only trigger if both agree.
    """
    h   = binary.shape[0]
    r0  = int(INTERSECTION_CHECK_ROW_START * h)
    r1  = int(INTERSECTION_CHECK_ROW_END   * h)
    hr  = _horizontal_energy_ratio(binary, r0, r1)
    co  = _center_occupancy(binary, r0, r1)
    return (hr > MAX_HORIZ_ENERGY_RATIO * 1.5) and (co > MAX_CENTER_OCC_RATIO * 1.5)


# -------------------------------------------------
# 13) Main lane detection
# -------------------------------------------------

def detect_lanes(
    mask: np.ndarray,
) -> Tuple[Optional[BoundaryFit], Optional[BoundaryFit], np.ndarray]:
    """Run the full lane detection on a BEV binary mask.

    Output: left fit (or None), right fit (or None), and the histogram.
    """
    MIN_PIX = 2500
    _, w = mask.shape
    empty_hist = np.zeros(w, dtype=np.float32)

    if int((mask > 0).sum()) < MIN_PIX:
        return None, None, empty_hist

    # Soft clutter guard: only suppress if the mask is dense enough
    # that an intersection-like pattern is plausible.
    if float((mask > 0).mean()) > 0.08 and is_heavy_clutter(mask):
        return None, None, empty_hist

    l_seeds, r_seeds, hist = find_seed_columns(mask)
    candidates: List[BoundaryFit] = []

    for s in l_seeds:
        bf = fit_boundary(mask, s, "left")
        if bf is not None and _check_mask_fill(mask, bf):
            candidates.append(bf)

    for s in r_seeds:
        bf = fit_boundary(mask, s, "right")
        if bf is not None and _check_mask_fill(mask, bf):
            candidates.append(bf)

    chosen = pick_ego_boundaries(candidates, mask.shape)

    if len(chosen) == 0:
        return None, None, hist
    if len(chosen) == 1:
        # Can happen when only one good boundary is detected.
        bf = chosen[0]
        return (bf, None, hist) if bf.side == "left" else (None, bf, hist)

    left  = next((b for b in chosen if b.side == "left"),  None)
    right = next((b for b in chosen if b.side == "right"), None)
    return left, right, hist


# -------------------------------------------------
# 14) Temporal smoothing
# -------------------------------------------------

def _smooth_col(prev: Optional[int], cur: int) -> int:
    if prev is None:
        return cur
    if abs(cur - prev) > LANE_MAX_JUMP_PX:
        return prev   # reject jump; keep previous
    return int(round(LANE_TRACK_ALPHA * prev + (1.0 - LANE_TRACK_ALPHA) * cur))


def update_tracker(
    state: TrackState,
    left: Optional[BoundaryFit],
    right: Optional[BoundaryFit],
) -> None:
    """Temporal smoothing for the lanes actually detected in the current frame.

    If only one boundary is detected, only that one is kept visible.
    The missing boundary is cleared, so the visualization matches the real detection.
    """
    new_left = int(round(left.x_near)) if left is not None else None
    new_right = int(round(right.x_near)) if right is not None else None

    # No lane detected in the current frame
    if new_left is None and new_right is None:
        state.lost_count += 1
        if state.lost_count > MAX_TRACK_LOST_FRAMES:
            state.left_col = None
            state.right_col = None
            state.lane_width_px = None
            state.center_col = None

            state.left_type = "unknown"
            state.right_type = "unknown"
            state.left_pending_type = None
            state.right_pending_type = None
            state.left_pending_count = 0
            state.right_pending_count = 0
        return

    # At least one lane is present in this frame
    state.lost_count = 0

    # Update left lane only if detected, otherwise clear it
    if new_left is not None:
        state.left_col = _smooth_col(state.left_col, new_left)
    else:
        state.left_col = None
        state.left_type = "unknown"
        state.left_pending_type = None
        state.left_pending_count = 0

    # Update right lane only if detected, otherwise clear it
    if new_right is not None:
        state.right_col = _smooth_col(state.right_col, new_right)
    else:
        state.right_col = None
        state.right_type = "unknown"
        state.right_pending_type = None
        state.right_pending_count = 0

    # Update lane width and center only if both boundaries are available
    if state.left_col is not None and state.right_col is not None:
        mw = float(state.right_col - state.left_col)
        mc = 0.5 * float(state.left_col + state.right_col)

        state.lane_width_px = (
            mw if state.lane_width_px is None
            else 0.88 * state.lane_width_px + 0.12 * mw
        )
        state.center_col = (
            mc if state.center_col is None
            else 0.88 * state.center_col + 0.12 * mc
        )
    else:
        state.lane_width_px = None
        state.center_col = None


def _update_stable_type(
    stable: str,
    pending: Optional[str],
    pending_count: int,
    raw: str,
) -> Tuple[str, Optional[str], int]:
    """Small hysteresis for lane type labels.
    """
    if raw == "unknown":
        return stable, None, 0

    if stable == "unknown":
        return raw, None, 0

    if raw == stable:
        return stable, None, 0

    # Candidate switch
    if pending == raw:
        pending_count += 1
    else:
        pending = raw
        pending_count = 1

    if pending_count >= TYPE_CONFIRM_FRAMES:
        return pending, None, 0
    return stable, pending, pending_count


def update_lane_type_tracker(
    state: TrackState,
    lane_mask: np.ndarray,
    left_col: Optional[int],
    right_col: Optional[int],
) -> List[str]:
    """Compute and stabilize lane types for the current frame."""
    out: List[str] = []

    if left_col is not None:
        raw = classify_lane_type(lane_mask, int(left_col))
        state.left_type, state.left_pending_type, state.left_pending_count = _update_stable_type(
            state.left_type, state.left_pending_type, state.left_pending_count, raw
        )
        out.append(state.left_type)
    else:
        state.left_type = "unknown"
        state.left_pending_type = None
        state.left_pending_count = 0

    if right_col is not None:
        raw = classify_lane_type(lane_mask, int(right_col))
        state.right_type, state.right_pending_type, state.right_pending_count = _update_stable_type(
            state.right_type, state.right_pending_type, state.right_pending_count, raw
        )
        out.append(state.right_type)
    else:
        state.right_type = "unknown"
        state.right_pending_type = None
        state.right_pending_count = 0

    return out


# -------------------------------------------------
# 15) Lane type classification (solid vs dashed)
# -------------------------------------------------

def classify_lane_type(binary: np.ndarray, col: int, half: int = 10) -> str:
    """Classify a boundary column as 'continuous', 'dashed', or 'unknown'."""
    if not CLASSIFY_TYPE:
        return "unknown"

    h, w  = binary.shape
    col   = int(np.clip(col, 0, w - 1))
    c1, c2 = max(0, col - half), min(w, col + half + 1)
    y0, y1 = int(0.15 * h), int(0.97 * h)

    strip = (binary[y0:y1, c1:c2] > 0).astype(np.uint8)
    if strip.size == 0:
        return "unknown"

    row_pres = (strip.sum(axis=1) >= max(2, int(0.18 * strip.shape[1]))).astype(np.uint8)
    # Close small gaps (helps with dashed markings)
    row_pres = cv2.morphologyEx(
        (row_pres[:, None] * 255), cv2.MORPH_CLOSE, np.ones((11, 1), np.uint8)
    ).ravel() > 0

    support_ratio = float(row_pres.mean())
    n_total       = len(row_pres)
    # Count runs
    runs = []
    cur  = 0
    for v in row_pres:
        if v:
            cur += 1
        else:
            if cur: runs.append(cur); cur = 0
    if cur: runs.append(cur)
    longest_run_ratio = max(runs) / max(n_total, 1) if runs else 0.0
    n_runs = len(runs)

    if support_ratio >= 0.60 and longest_run_ratio >= 0.40 and n_runs <= 3:
        return "continuous"
    if 0.10 <= support_ratio <= 0.58 and n_runs >= 3 and support_ratio >= 0.07:
        return "dashed"
    return "unknown"


# -------------------------------------------------
# 16) YOLO obstaclelet detection
# -------------------------------------------------

class ObstacleDetector:
    def __init__(
        self,
        model_path: str  = "yolov8n.pt",
        conf:       float = 0.35,
        device:     str   = "cpu",
    ) -> None:
        self.conf   = conf
        self.device = device
        self.model  = None
        self._cpu_fallback = False

        if YOLO_AVAILABLE:
            self.model = YOLO(model_path)
        else:
            print("[WARN] ultralytics not installed — obstacle detection disabled.")

    def _predict(self, image: np.ndarray):
        try:
            return self.model.predict(
                source=image, conf=self.conf, verbose=False, device=self.device
            )
        except Exception as exc:
            if self.device == "cuda" and not self._cpu_fallback:
                print(f"[WARN] CUDA failed ({exc}). Falling back to CPU.")
                self.device, self._cpu_fallback = "cpu", True
                return self.model.predict(
                    source=image, conf=self.conf, verbose=False, device="cpu"
                )
            raise

    def detect(
        self,
        image:     np.ndarray,
        left_col:  Optional[int],
        right_col: Optional[int],
    ) -> List[Obstacle]:
        if self.model is None:
            return []

        results = self._predict(image)
        if not results or results[0].boxes is None:
            return []

        res   = results[0]
        names = res.names
        have_pair = (left_col is not None and right_col is not None)
        out: List[Obstacle] = []

        for box in res.boxes:
            cls_id = int(box.cls.item())
            label  = names.get(cls_id, str(cls_id))
            if label not in YOLO_TRACKED_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].detach().cpu().numpy().astype(int))
            foot_y  = float(y2)
            foot_x  = float((x1 + x2) / 2.0)

            dist_raw = image_row_to_distance(foot_y)
            dist = None if dist_raw is None else max(0.0, float(dist_raw) * OBSTACLE_DISTANCE_SCALE)

            in_lane = False
            # I keep the x-projection based on the raw geometry (unscaled).
            # Only the *reported* distance is scaled.
            if have_pair and dist_raw is not None:
                xw      = image_u_to_world_x(foot_x, float(dist_raw))
                bev_col = world_x_to_bev_col(xw)
                cL      = min(left_col, right_col)  # type: ignore[arg-type]
                cR      = max(left_col, right_col)  # type: ignore[arg-type]
                in_lane = cL <= bev_col <= cR

            out.append(Obstacle(
                label=label, confidence=float(box.conf.item()),
                bbox=(x1, y1, x2, y2),
                distance_m=dist, in_lane=in_lane,
            ))

        return out


def select_display_obstacles(
    detections: List[Obstacle],
    left_col:   Optional[int],
    right_col:  Optional[int],
) -> List[Obstacle]:
    if left_col is not None and right_col is not None:
        return [d for d in detections if d.in_lane]
    return detections


# -------------------------------------------------
# 17) Drawing
# -------------------------------------------------

def _txt(img: np.ndarray, text: str, org: Tuple[int, int],
         color: Tuple[int, int, int], scale: float = 0.65, thick: int = 2) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (10, 10, 10), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color,        thick,     cv2.LINE_AA)


def _draw_vertical_marker(out: np.ndarray, col: int, lane_type: str) -> None:
    col = int(np.clip(col, 0, out.shape[1] - 1))
    H   = out.shape[0]
    GREEN = (0, 255, 0)
    if lane_type == "dashed":
        seg, gap, y = 26, 18, 0
        while y < H - 1:
            y2 = min(H - 1, y + seg)
            cv2.line(out, (col, y), (col, y2), GREEN, 2, cv2.LINE_AA)
            y += seg + gap
    else:
        cv2.line(out, (col, 0), (col, H - 1), GREEN, 2, cv2.LINE_AA)


def draw_bev_result(
    bev_img:   np.ndarray,
    left_col:  Optional[int],
    right_col: Optional[int],
    lane_types: List[str],
) -> np.ndarray:
    out = bev_img.copy()

    if left_col is None and right_col is None:
        _txt(out, "No lanes found",
             (BEV_W // 2 - 110, BEV_H // 2), (0, 0, 255), 0.9, 2)
        return out

    pairs: List[Tuple[int, str]] = []
    if left_col  is not None:
        pairs.append((int(left_col),  lane_types[0] if len(lane_types) > 0 else "unknown"))
    if right_col is not None:
        idx = 1 if left_col is not None else 0
        pairs.append((int(right_col), lane_types[idx] if len(lane_types) > idx else "unknown"))

    for i, (col, typ) in enumerate(pairs):
        _draw_vertical_marker(out, col, typ)
        xw  = col_to_world_x(col)
        lbl = f"{typ}  {xw:+.2f} m" if CLASSIFY_TYPE else f"{xw:+.2f} m"
        _txt(out, lbl, (max(8, col - 65), 28 + 26 * i), (0, 255, 0), 0.52, 1)

    return out


def _dashed_poly(canvas: np.ndarray, pts: List[Tuple[int, int]],
                 color: Tuple[int, int, int], thick: int,
                 on: int = 7, off: int = 5) -> None:
    """Draw a dashed polyline (simple on/off pattern along the segments)."""
    period = max(on + off, 1)
    for i in range(len(pts) - 1):
        if (i % period) < on:
            cv2.line(canvas, pts[i], pts[i + 1], color, thick, cv2.LINE_AA)


def draw_overlay_on_original(
    img:        np.ndarray,
    left_col:   Optional[int],
    right_col:  Optional[int],
    lane_types: List[str],
    H_bev_to_img: Optional[np.ndarray],
) -> np.ndarray:
    out = img.copy()

    if left_col is None and right_col is None:
        _txt(out, "No lanes found", (50, IMG_H // 2), (0, 0, 255), 1.0, 2)
        return out

    cols: List[Tuple[int, str]] = []
    if left_col  is not None:
        cols.append((int(left_col),  lane_types[0] if len(lane_types) > 0 else "unknown"))
    if right_col is not None:
        idx = 1 if left_col is not None else 0
        cols.append((int(right_col), lane_types[idx] if len(lane_types) > idx else "unknown"))

    for col, typ in cols:
        pts: List[Tuple[int, int]] = []

        xw = col_to_world_x(col)
        for yw in np.linspace(Y_MIN_M, Y_MAX_M, 180):
            uv = world_to_image(xw, float(yw))
            if uv is None:
                continue
            u, v = uv
            if 0 <= u < IMG_W and 0 <= v < IMG_H:
                pts.append((u, v))

        if len(pts) < 2:
            continue

        GREEN = (0, 255, 0)
        if typ == "dashed":
            _dashed_poly(out, pts, GREEN, 4)
        else:
            cv2.polylines(out, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)],
                          False, GREEN, 4, cv2.LINE_AA)

    return out


def draw_obstacles_on_original(img: np.ndarray, dets: List[Obstacle]) -> np.ndarray:
    out = img.copy()
    for det in dets:
        x1, y1, x2, y2 = det.bbox
        lbl = f"{det.label} {det.distance_m:.1f}m" if det.distance_m is not None else det.label
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        ty = y1 - 10 if y1 > 30 else y1 + 25
        _txt(out, lbl, (x1, ty), (0, 0, 255), 0.65, 2)
    return out


def draw_obstacles_on_bev(
    bev: np.ndarray,
    dets: List[Obstacle],
    H_bev_to_img: Optional[np.ndarray],
) -> np.ndarray:
    out = bev.copy()
    RED = (0, 0, 255)

    H_img_to_bev: Optional[np.ndarray] = None
    if H_bev_to_img is not None:
        try:
            H_img_to_bev = np.linalg.inv(H_bev_to_img)
        except Exception:
            H_img_to_bev = None

    def _bev_length_for_label(label: str) -> float:
        # Rough footprint length (meters) on the ground plane
        if label in {"bus", "truck", "train"}:
            return 6.5
        if label in {"car"}:
            return 4.5
        if label in {"motorcycle", "bicycle"}:
            return 2.2
        if label in {"person"}:
            return 1.0
        return 3.0

    for det in dets:
        x1, y1, x2, y2 = det.bbox

        if H_img_to_bev is not None:
            # Project the 2D bbox corners into BEV using the inverse homography.
            # This makes the BEV rectangle line up with the BEV we render.
            pts_img = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            pts_bev = cv2.perspectiveTransform(pts_img, H_img_to_bev).reshape(-1, 2)
            xs = pts_bev[:, 0]
            ys = pts_bev[:, 1]

            # If the obstacle ends up completely outside the BEV canvas, skip it.
            # Requirement: if it doesn't show up in BEV, don't mark it in BEV.
            if (
                xs.max() < 0
                or xs.min() > (out.shape[1] - 1)
                or ys.max() < 0
                or ys.min() > (out.shape[0] - 1)
            ):
                continue

            x_min = int(np.clip(np.floor(xs.min()), 0, out.shape[1] - 1))
            x_max = int(np.clip(np.ceil(xs.max()),  0, out.shape[1] - 1))
            y_min = int(np.clip(np.floor(ys.min()), 0, out.shape[0] - 1))
            y_max = int(np.clip(np.ceil(ys.max()),  0, out.shape[0] - 1))

            if x_max > x_min and y_max > y_min:
                cv2.rectangle(out, (x_min, y_min), (x_max, y_max), RED, 2)
                label_dist = f"{det.label} {det.distance_m:.1f}m" if det.distance_m is not None else det.label
                tx = int(np.clip(x_max + 10, 0, out.shape[1] - 160))
                ty = int(np.clip(y_min + 16, 16, out.shape[0] - 1))
                _txt(out, label_dist, (tx, ty), (240, 240, 240), 0.50, 1)
            continue

        # the estimated distance + bbox lateral span.
        if det.distance_m is None:
            continue

        # If the distance is outside the BEV forward range, skip.
        dist_raw = float(det.distance_m)
        if not (Y_MIN_M <= dist_raw <= Y_MAX_M):
            continue
        dist = dist_raw
        xw_l = float(image_u_to_world_x(float(x1), dist))
        xw_r = float(image_u_to_world_x(float(x2), dist))
        if xw_l > xw_r:
            xw_l, xw_r = xw_r, xw_l
        pad_x = 0.15
        xw_l -= pad_x
        xw_r += pad_x
        length_m = _bev_length_for_label(det.label)
        y0 = float(np.clip(dist, Y_MIN_M, Y_MAX_M))
        y1w = float(np.clip(dist + length_m, Y_MIN_M, Y_MAX_M))
        c0 = world_x_to_bev_col(xw_l)
        c1 = world_x_to_bev_col(xw_r)
        r0 = world_y_to_bev_row(y0)
        r1 = world_y_to_bev_row(y1w)
        x_min, x_max = (min(c0, c1), max(c0, c1))
        y_min, y_max = (min(r0, r1), max(r0, r1))
        cv2.rectangle(out, (x_min, y_min), (x_max, y_max), RED, 2)
        tx = int(np.clip(x_max + 10, 0, out.shape[1] - 160))
        ty = int(np.clip(y_min + 16, 16, out.shape[0] - 1))
        _txt(out, f"{det.label} {det.distance_m:.1f}m", (tx, ty), (240, 240, 240), 0.50, 1)
    return out


# -------------------------------------------------
# 18) Per-frame pipeline
# -------------------------------------------------

def process_image(
    path:              str,
    map_x:             np.ndarray,
    map_y:             np.ndarray,
    valid_mask:        np.ndarray,
    H_bev_to_img:      Optional[np.ndarray],
    save_dir:          Optional[str],
    wait_ms:           int,
    obstacle_detector: Optional[ObstacleDetector],
    show_mask:         bool,
) -> int:
    img = cv2.imread(path)
    if img is None:
        print(f"[skip] cannot read {path}")
        return -1

    if img.shape[:2] != (IMG_H, IMG_W):
        img = cv2.resize(img, (IMG_W, IMG_H))

    # ── BEV warp ──────────────────────────────────────────
    bev = warp_to_bev(img, map_x, map_y, H_bev_to_img)

    # ── Preprocessing + segmentation ─────────────────────
    _, features, lane_mask = segment_lane_pixels(bev, valid_mask)

    # ── Lane detection ───────────────────────────────────
    left_bf, right_bf, hist = detect_lanes(lane_mask)

    # ── Temporal smoothing ───────────────────────────────
    update_tracker(TRACK, left_bf, right_bf)

    # Use the smoothed columns for display + YOLO corridor checks
    left_col  = TRACK.left_col
    right_col = TRACK.right_col

    # ── Lane type classification ─────────────────────────
    lane_types = update_lane_type_tracker(TRACK, lane_mask, left_col, right_col)

    # ── Rendering ────────────────────────────────────────
    bev_out  = draw_bev_result(bev, left_col, right_col, lane_types)
    orig_out = draw_overlay_on_original(img, left_col, right_col, lane_types, H_bev_to_img)

    # ── Obstacle detection ───────────────────────────────
    if obstacle_detector is not None:
        dets = obstacle_detector.detect(img, left_col, right_col)
        dets = select_display_obstacles(dets, left_col, right_col)
        if dets:
            orig_out = draw_obstacles_on_original(orig_out, dets)
            bev_out  = draw_obstacles_on_bev(bev_out, dets, H_bev_to_img)

    # ── Console log ──────────────────────────────────────
    name = os.path.basename(path)
    left_s  = "yes" if left_col is not None else "none"
    right_s = "yes" if right_col is not None else "none"
    print(f"{name} | left={left_s} | right={right_s}")

    # ── Display ───────────────────────────────────────────
    target_h  = 650
    orig_show = cv2.resize(
        orig_out,
        (int(orig_out.shape[1] * target_h / orig_out.shape[0]), target_h),
        interpolation=cv2.INTER_LINEAR,
    )
    bev_show  = cv2.resize(
        bev_out,
        (int(bev_out.shape[1] * target_h / bev_out.shape[0]), target_h),
        interpolation=cv2.INTER_LINEAR,
    )
    cv2.imshow("GOLD lane detection", np.hstack([orig_show, bev_show]))

    if show_mask:
        bw = (lane_mask > 0).astype(np.uint8) * 255
        bin_show = cv2.resize(
            bw,
            (int(bw.shape[1] * target_h / bw.shape[0]), target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("BEV binary mask", bin_show)

    # ── Optional save ─────────────────────────────────────
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.splitext(name)[0]
        cv2.imwrite(os.path.join(save_dir, f"orig_{stem}.jpg"), orig_out)
        cv2.imwrite(os.path.join(save_dir, f"bev_{stem}.jpg"),  bev_out)
        cv2.imwrite(os.path.join(save_dir, f"bin_{stem}.jpg"),  lane_mask)

    return cv2.waitKey(max(1, wait_ms)) & 0xFF


# -------------------------------------------------
# 19) Input expansion
# -------------------------------------------------

def expand_inputs(inp: str) -> List[str]:
    inp = inp.strip().strip('"').strip("'")
    if os.path.isdir(inp):
        files: List[str] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files.extend(glob.glob(os.path.join(inp, ext)))
        return sorted(set(files))
    if any(ch in inp for ch in ["*", "?", "["]):
        return sorted(glob.glob(inp))
    return [inp]


# -------------------------------------------------
# 20) Entry point
# -------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GOLD lane detection for PandaSet — improved pipeline"
    )
    parser.add_argument(
        "input",
        help="Directory, glob, or image path (e.g. '.../front_camera/')",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Show the BEV binary mask window",
    )

    args = parser.parse_args()

    files = [f for f in expand_inputs(args.input) if os.path.isfile(f)]
    if not files:
        raise SystemExit(f"No images found for input: {args.input}")

    # Build IPM tables once (same for all frames)
    map_x, map_y   = build_ipm_maps()
    valid_mask      = build_valid_mask(map_x, map_y)
    H_bev_to_img   = compute_homography_bev_to_img()

    # YOLO is always enabled in this script.
    if not YOLO_AVAILABLE:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install ultralytics"
        )
    detector = ObstacleDetector()

    print(f"Found {len(files)} images")
    print(f"BEV size: {BEV_W}×{BEV_H}  ({PIXELS_PER_METER} px/m)")
    print("Press Q to quit, any other key to advance")

    for path in files:
        key = process_image(
            path, map_x, map_y, valid_mask, H_bev_to_img,
            None, 0, detector, bool(args.mask),
        )
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()