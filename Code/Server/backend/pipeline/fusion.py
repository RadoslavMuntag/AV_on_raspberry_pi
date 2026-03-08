from __future__ import annotations

import time
from typing import Optional

from .config import PipelineConfig
from ..contracts import PerceptionFrame, WorldState, SensorType

class FusionModule:
    def __init__(self, cfg: Optional[PipelineConfig] = None) -> None:
        self.cfg = cfg or PipelineConfig()

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