from __future__ import annotations

import time

from .config import PipelineConfig
from ..contracts import PerceptionFrame, InfraredState
from ..services.hardware import VehicleHardware
from .vision import detect_line_geometry

class PerceptionModule:
    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self.cfg = cfg or PipelineConfig()

    def _decode_infrared_line(self, raw: InfraredState | None) -> tuple[float | None, float]:
        """
        Decode line position from a 3-sensor bit pattern.
        Returns (line_error, confidence).
        """
        if raw is None:
            return None, 0.0

        # Example weight model for 3 sensors (right->left) 
        weights = [-1.0, 0.0, 1.0]  # IR1=right, IR2=center, IR3=left
        bits = [(raw >> i) & 1 for i in range(3)]

        if __name__ == "__main__":
            for i, b in enumerate(bits):
                # Debug print for sensor states only when running this module directly
                print(f"IR{i+1}: {'ON' if b else 'OFF'}") 

        active = [i for i, b in enumerate(bits) if b == 1]
        if not active:
            return None, 0.0

        err = sum(weights[i] for i in active) / len(active)
        conf = min(1.0, len(active) / 3.0)
        return err / 2.0, conf  # normalize roughly into [-1, 1]

    def read(self, hardware: VehicleHardware) -> PerceptionFrame:
        ts = time.monotonic()
        faults: list[str] = []

        ultrasonic = hardware.read_ultrasonic()
        if ultrasonic is None:
            faults.append("ultrasonic_unavailable")

        # infrared_raw_int = hardware.read_infrared()
        # infrared_raw: InfraredState | None = None
        # if infrared_raw_int is None:
        #     faults.append("infrared_unavailable")
        # else:
        #     try:
        #         infrared_raw = InfraredState(infrared_raw_int)
        #     except ValueError:
        #         faults.append(f"infrared_invalid_value:{infrared_raw_int}")

        # Infrared fallback
        #ir_line_error, ir_line_conf = self._decode_infrared_line(infrared_raw)
        left_distance = hardware.read_left_encoder()
        right_distance = hardware.read_right_encoder()

        camera_frame = hardware.get_usb_jpeg_frame()
        cam_offset: float | None = None
        cam_angle: float | None = None
        cam_curvature: float | None = None

        angle, curvature, offset = None, None, None
        if camera_frame is None:
            faults.append("camera_unavailable")
        else:
            try:
                cam_angle, cam_curvature, cam_offset, debug = detect_line_geometry(camera_frame)
                hardware.set_debug_frame(debug)
            except Exception as e:
                faults.append(f"line_geometryppp_error: {str(e)}")

            angle = cam_angle
            curvature = cam_curvature
            offset = cam_offset

        return PerceptionFrame(
            ts=ts,
            ultrasonic_cm=ultrasonic,
            left_encoder_cm=left_distance,
            right_encoder_cm=right_distance,

            line_angle=angle,
            line_curvature=curvature,
            line_offset=offset,
            camera_ok=hardware.ready,
            faults=faults,
        )