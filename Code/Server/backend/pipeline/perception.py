from __future__ import annotations

import time

from .config import PipelineConfig
from ..contracts import PerceptionFrame, InfraredState
from ..services.hardware import VehicleHardware
from .vision import detect_line_geometry

class PerceptionModule:
    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self.cfg = cfg or PipelineConfig()

    def read(self, hardware: VehicleHardware, dt: float) -> PerceptionFrame:
        ts = time.monotonic()
        faults: list[str] = []

        ultrasonic = hardware.read_ultrasonic()
        if ultrasonic is None:
            faults.append("ultrasonic_unavailable")

        left_speed = hardware.read_left_encoder(self.cfg.wheel_radius, dt)
        if left_speed is None:
            faults.append("left_encoder_unavailable")
        right_speed = hardware.read_right_encoder(self.cfg.wheel_radius, dt)
        if right_speed is None:
            faults.append("right_encoder_unavailable")

        camera_frame = hardware.get_jpeg_frame()
        cam_offset: float | None = None
        cam_angle: float | None = None
        cam_curvature: float | None = None
        cam_confidence: float = 0.0

        angle, curvature, offset, confidence = None, None, None, 0.0
        if camera_frame is None:
            faults.append("camera_unavailable")
        else:
            try:
                cam_angle, cam_curvature, cam_offset, cam_confidence, debug = detect_line_geometry(camera_frame)
                hardware.set_debug_frame(debug)
            except Exception as e:
                faults.append(f"line_geometryppp_error: {str(e)}")

            if cam_confidence >= self.cfg.min_confidence:    
                angle = cam_angle
                curvature = cam_curvature
                offset = cam_offset
            confidence = cam_confidence
        
        if self.cfg.DEBUG:
            print("DEBUG: Perception read - ultrasonic:", ultrasonic, "left_speed:", left_speed, "right_speed:", right_speed, "angle:", angle, "curvature:", curvature, "offset:", offset, "faults:", faults)

        return PerceptionFrame(
            ts=ts,
            ultrasonic_cm=ultrasonic,
            left_speed=left_speed,
            right_speed=right_speed,

            line_angle=angle,
            line_curvature=curvature,
            line_offset=offset,
            line_confidence=confidence,
            camera_ok=hardware.ready,
            faults=faults,
        )