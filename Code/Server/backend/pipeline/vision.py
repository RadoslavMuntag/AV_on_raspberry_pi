from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

def detect_line_error_from_jpeg(
    jpeg_bytes: bytes,
    canny_low: int = 60,
    canny_high: int = 150,
    hough_threshold: int = 30,
    min_line_length: int = 25,
    max_line_gap: int = 20,
) -> tuple[Optional[float], float]:
    """
    Returns:
      line_error: normalized in [-1, 1], negative=left, positive=right
      confidence: [0, 1]
    """
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, 0.0

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    # ROI: lower half only
    mask = np.zeros_like(edges)
    roi = np.array([[
        (0, h),
        (w, h),
        (int(0.65 * w), int(0.55 * h)),
        (int(0.35 * w), int(0.55 * h)),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi, 255)
    roi_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None or len(lines) == 0:
        return None, 0.0

    # Robust center estimate from detected segments
    x_samples: list[float] = []
    y_ref = h - 10

    for l in lines[:, 0]:
        x1, y1, x2, y2 = map(float, l)
        dx = x2 - x1
        dy = y2 - y1

        # Ignore near-horizontal noise
        if abs(dy) < 5:
            continue

        # Interpolate x at y_ref on this segment's line
        t = (y_ref - y1) / (dy if dy != 0 else 1e-6)
        x_at_ref = x1 + t * dx
        if 0 <= x_at_ref <= w:
            x_samples.append(x_at_ref)

    if not x_samples:
        return None, 0.0

    lane_x = float(np.median(x_samples))
    center_x = w / 2.0
    err = (lane_x - center_x) / (w / 2.0)  # [-1, 1] approx
    err = max(-1.0, min(1.0, err))

    confidence = min(1.0, len(x_samples) / 12.0)
    return err, confidence