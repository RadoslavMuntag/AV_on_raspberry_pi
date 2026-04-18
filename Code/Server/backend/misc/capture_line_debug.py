from __future__ import annotations
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedCallResult=false, reportAny=false, reportImplicitStringConcatenation=false

import argparse
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import cv2
from cv2.typing import MatLike
import numpy as np

from backend.misc.usb_camera import USBCamera

CameraBackend = Literal["usb", "picamera"]


def detect_line_geometry_with_stages(
    jpeg_bytes: bytes,
) -> tuple[float | None, float | None, float | None, float, dict[str, MatLike]]:
    """
    Copy of the contour-based line geometry detector with stage snapshots.

    Returns:
        angle (rad)      : heading direction of the line
        curvature        : curvature of fitted polynomial
        offset           : horizontal offset from image center in range [-1, 1]
        confidence       : line detection confidence in range [0, 1]
        stages           : dict of named snapshot images
    """

    stages: dict[str, MatLike] = {}

    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode JPEG bytes into an image")

    stages["01_raw"] = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stages["02_gray"] = gray.copy()

    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    stages["03_threshold_inv"] = binary.copy()

    h, w = binary.shape
    mask = np.zeros_like(binary)
    mask[int(h * 0.45) : h, 0:w] = 255
    stages["04_roi_mask"] = mask.copy()

    binary = cv2.bitwise_and(binary, binary, mask=mask)
    stages["05_threshold_roi"] = binary.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    stages["06_morph_open"] = binary.copy()

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours_debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if contours:
        _ = cv2.drawContours(contours_debug, contours, -1, (0, 255, 255), 1)
    stages["07_all_contours"] = contours_debug

    if len(contours) == 0:
        no_line_debug = frame.copy()
        _ = cv2.putText(
            no_line_debug,
            "No contour found",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        stages["08_final_debug"] = no_line_debug
        return None, None, None, 0.0, stages

    contour = max(contours, key=cv2.contourArea)

    largest_debug = frame.copy()
    _ = cv2.drawContours(largest_debug, [contour], -1, (255, 0, 0), 2)
    stages["08_largest_contour"] = largest_debug

    points = contour[:, 0, :]
    if len(points) < 3:
        few_points_debug = largest_debug.copy()
        _ = cv2.putText(
            few_points_debug,
            "Largest contour has too few points",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        stages["09_final_debug"] = few_points_debug
        return None, None, None, 0.0, stages

    x = points[:, 0]
    y = points[:, 1]

    coeffs = np.polyfit(y, x, 2)
    a, b, c = coeffs

    y_eval = frame.shape[0]
    slope = 2 * a * y_eval + b
    angle = float(np.arctan(slope))

    curvature = float(abs(2 * a) / ((1 + slope**2) ** 1.5))

    x_line = a * y_eval**2 + b * y_eval + c
    center = frame.shape[1] / 2
    offset = float(np.clip((x_line - center) / center, -1.0, 1.0))

    frame_area = float(h * w)
    contour_area = float(cv2.contourArea(contour))
    area_score = float(np.clip(contour_area / (0.12 * frame_area), 0.0, 1.0))

    y_span = float(y.max() - y.min())
    vertical_coverage = float(np.clip(y_span / max(1.0, float(h)), 0.0, 1.0))

    x_fit = a * y**2 + b * y + c
    rmse = float(np.sqrt(np.mean((x - x_fit) ** 2)))
    fit_score = float(np.clip(1.0 - (rmse / max(1.0, 0.2 * w)), 0.0, 1.0))

    point_score = float(np.clip(len(points) / 250.0, 0.0, 1.0))
    confidence = float(
        np.clip(
            0.35 * area_score + 0.30 * vertical_coverage + 0.25 * fit_score + 0.10 * point_score,
            0.0,
            1.0,
        )
    )

    debug = frame.copy()
    for yi in range(0, frame.shape[0], 5):
        xi = int(a * yi**2 + b * yi + c)
        if 0 <= xi < frame.shape[1]:
            _ = cv2.circle(debug, (xi, yi), 2, (0, 255, 0), -1)

    _ = cv2.drawContours(debug, [contour], -1, (255, 0, 0), 2)
    _ = cv2.putText(
        debug,
        f"slope: {float(slope):.3f} angle: {angle:.3f} conf: {confidence:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    stages["09_final_debug"] = debug
    return angle, curvature, offset, confidence, stages


def capture_single_jpeg(
    camera_backend: CameraBackend,
    usb_device: int,
    width: int,
    height: int,
    timeout_sec: float,
) -> bytes:
    if camera_backend == "picamera":
        from model.sensors.camera import Camera

        camera = Camera(preview_size=(width, height), stream_size=(width, height))
        try:
            camera.start_stream()
            frame = cast(bytes | None, camera.get_frame())
            if frame is None:
                raise TimeoutError("No frame received from Picamera stream")
            return frame
        finally:
            camera.stop_stream()
            camera.close()

    camera = USBCamera(device=usb_device, preview_size=(width, height), stream_size=(width, height), fps=20)
    try:
        camera.start_stream()
        frame = camera.get_frame(timeout=timeout_sec)
        if frame is None:
            raise TimeoutError(f"No frame received within {timeout_sec:.1f}s")
        return frame
    finally:
        camera.stop_stream()
        camera.close()


def save_stages(stages: dict[str, MatLike], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, image in stages.items():
        target = out_dir / f"{name}.jpg"
        ok = cv2.imwrite(str(target), image)
        if not ok:
            raise RuntimeError(f"Failed to save stage image: {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture one camera frame and save line-detection pipeline snapshots.",
    )
    _ = parser.add_argument(
        "--camera-backend",
        choices=["usb", "picamera"],
        default="picamera",
        help="Camera backend to use (default: picamera)",
    )
    _ = parser.add_argument("--usb-device", type=int, default=8, help="V4L2 USB camera device index (default: 8)")
    _ = parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    _ = parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    _ = parser.add_argument("--timeout", type=float, default=3.0, help="Frame wait timeout in seconds (default: 3.0)")
    _ = parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "snapshots",
        help="Base output directory. A timestamped folder is created inside it.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cast(Path, args.out_dir) / timestamp

    jpeg_bytes = capture_single_jpeg(
        camera_backend=cast(CameraBackend, args.camera_backend),
        usb_device=cast(int, args.usb_device),
        width=cast(int, args.width),
        height=cast(int, args.height),
        timeout_sec=cast(float, args.timeout),
    )

    angle, curvature, offset, confidence, stages = detect_line_geometry_with_stages(jpeg_bytes)
    save_stages(stages, run_dir)

    print(f"Saved {len(stages)} snapshots to: {run_dir}")
    print(f"Results -> angle={angle}, curvature={curvature}, offset={offset}, confidence={confidence:.3f}")


if __name__ == "__main__":
    main()
