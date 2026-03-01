from __future__ import annotations

import time
from typing import Optional

from .config import PerceptionConfig
from ..contracts import PerceptionFrame, InfraredState
from ..services.hardware import VehicleHardware


class PerceptionModule:
    def __init__(self, cfg: Optional[PerceptionConfig] = None) -> None:
        self.cfg = cfg or PerceptionConfig()

    def _decode_infrared_line(self, raw: Optional[InfraredState]) -> tuple[Optional[float], float]:
        """
        Decode line position from a 3-sensor bit pattern.
        Returns (line_error, confidence).
        """
        if raw is None:
            return None, 0.0

        # Example weight model for 3 sensors (right->left) 
        weights = [1.0, 0.0, -1.0]  # IR1=right, IR2=center, IR3=left
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

    def read(self, hardware: VehicleHardware ) -> PerceptionFrame:
        ts = time.monotonic()
        faults: list[str] = []

        ultrasonic = hardware.read_ultrasonic()
        if ultrasonic is None:
            faults.append("ultrasonic_unavailable")

        infrared_raw = hardware.read_infrared()
        if infrared_raw is None:
            faults.append("infrared_unavailable")

        line_error, line_conf = self._decode_infrared_line(infrared_raw)

        return PerceptionFrame(
            ts=ts,
            ultrasonic_cm=ultrasonic,
            infrared_raw=infrared_raw,
            line_error=line_error,
            line_confidence=line_conf,
            camera_ok=hardware.ready,
            faults=faults,
        )

if __name__ == "__main__":
    import model.sensors.infrared as infrared_module

    perception = PerceptionModule()
    ir_sensor = infrared_module.Infrared()

    try:
        while True:
            frame = perception._decode_infrared_line(ir_sensor.read_all_infrared())
            print(frame)
            time.sleep(0.5)
    except KeyboardInterrupt:
        ir_sensor.close()