from __future__ import annotations

import cv2
from cv2.typing import MatLike
import numpy as np

def detect_line_error_from_jpeg(
    jpeg_bytes: bytes,
    canny_low: int = 60,
    canny_high: int = 150,
    hough_threshold: int = 30,
    min_line_length: int = 25,
    max_line_gap: int = 20,
) -> tuple[float | None, float]:
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

def detect_line_pose_from_jpeg(
    jpeg_bytes: bytes,
    roi_y_start: float = 0.35,      # use lower 65% of frame
    min_pixels_per_row: int = 6,
) -> tuple[float | None, float | None, float]:
    """
    Returns:
      lateral_error: normalized [-1, 1], negative=left, positive=right
      heading_error: normalized [-1, 1], negative=line points left, positive=right
      confidence: [0, 1]
    """
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None, 0.0

    h, w = frame.shape[:2]
    y0 = int(max(0, min(h - 1, roi_y_start * h))) # 
    roi = frame[y0:h, :]

    # 1) Segment line (choose one strategy depending on your line color)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # For bright line on dark floor:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If your line is dark on bright floor, invert:
    # _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2) Morphology for cleanup
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    # 3) Scanlines from near (bottom) to far (top of ROI)
    ys = np.linspace(bw.shape[0] - 5, 5, num=12).astype(int)
    pts_x = []
    pts_y = []

    for y in ys:
        xs = np.where(bw[y, :] > 0)[0]
        if xs.size >= min_pixels_per_row:
            x_center = float(np.median(xs))
            pts_x.append(x_center)
            pts_y.append(float(y))

    if len(pts_x) < 3:
        return None, None, 0.0

    # 4) Fit x(y) = a*y + b
    a, b = np.polyfit(np.array(pts_y), np.array(pts_x), 1)

    # near point in ROI coordinates
    y_near = float(bw.shape[0] - 1)
    x_near = a * y_near + b

    # convert to full-frame x (ROI starts at y0, x unchanged)
    center_x = w / 2.0
    lateral_error = (x_near - center_x) / (w / 2.0)
    lateral_error = float(np.clip(lateral_error, -1.0, 1.0))

    # heading proxy from slope (scale for normalization)
    heading_error = float(np.clip(a * 0.8, -1.0, 1.0))

    # confidence from valid scanlines + mask density near bottom half
    valid_ratio = len(pts_x) / len(ys)
    density = float((bw[bw.shape[0] // 2 :, :] > 0).mean())
    confidence = float(np.clip(0.75 * valid_ratio + 0.25 * min(1.0, density * 8.0), 0.0, 1.0))

    return lateral_error, heading_error, confidence

def detect_line_geometry(jpeg_bytes: bytes) -> tuple[float | None, float | None, float | None, MatLike]:
    """
    Detect line direction and curvature from a top-down camera frame.

    Returns:
        angle (rad)      : heading direction of the line
        curvature        : curvature of fitted polynomial
        offset           : horizontal offset from image center in range [-1, 1]
        debug_image      : visualization image
    """

    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Failed to decode JPEG bytes into an image")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV) 

    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return None, None, None, frame 

    # Use the largest contour as the line
    contour = max(contours, key=cv2.contourArea)

    points = contour[:,0,:]

    x = points[:,0]
    y = points[:,1]

    # Fit polynomial (x as function of y)
    coeffs = np.polyfit(y, x, 2)
    a, b, c = coeffs

    # Heading angle (slope at bottom of image)
    y_eval = frame.shape[0]
    slope = 2*a*y_eval + b
    angle = np.arctan(slope)

    # Curvature calculation
    curvature = abs(2*a) / ((1 + slope**2)**1.5)

    # Offset from image center
    x_line = a*y_eval**2 + b*y_eval + c
    center = frame.shape[1] / 2
    offset = (x_line - center) / center

    # Debug visualization
    debug = frame.copy()

    for yi in range(0, frame.shape[0], 5):
        xi = int(a * yi**2 + b * yi + c)
        if 0 <= xi < frame.shape[1]:
            _ = cv2.circle(debug, (xi, yi), 2, (0, 255, 0), -1)

    _ = cv2.drawContours(debug, [contour], -1, (255,0,0), 2)

    return angle, curvature, offset, debug


    