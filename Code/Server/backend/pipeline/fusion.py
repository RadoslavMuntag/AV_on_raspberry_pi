from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

from ..contracts import PerceptionFrame, WorldState, SensorType


@dataclass(slots=True)
class FusionConfig:
    obstacle_threshold_cm: float = 25.0
    max_sensor_age_s: float = 0.25


class FusionModule:
    def __init__(self, cfg: Optional[FusionConfig] = None) -> None:
        self.cfg = cfg or FusionConfig()

    def fuse(self, p: PerceptionFrame) -> WorldState:
        ts = time.monotonic()
        dist = p.ultrasonic_cm
        obstacle = dist is not None and dist <= self.cfg.obstacle_threshold_cm
        stale = (ts - p.ts) > self.cfg.max_sensor_age_s

        return WorldState(
            ts=ts,
            obstacle_ahead=bool(obstacle),
            obstacle_distance_cm=dist,
            lane_detected=p.line_error is not None and p.line_confidence > 0.0,
            lateral_error=float(p.line_error or 0.0),
            lateral_confidence=float(p.line_confidence),
            sensor_health={
                SensorType.ULTRASONIC: p.ultrasonic_cm is not None,
                SensorType.INFRARED: p.infrared_raw is not None,
                SensorType.CAMERA: p.camera_ok,
            },
            stale=stale,
        )