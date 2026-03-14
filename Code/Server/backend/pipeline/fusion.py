from __future__ import annotations

import time

from .config import PipelineConfig
from ..contracts import PerceptionFrame, WorldState, SensorType

class FusionModule:
    def __init__(self, cfg: PipelineConfig | None = None) -> None:
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
            lane_detected=(p.line_offset is not None and p.line_angle is not None),
            line_offset=float(p.line_offset or 0.0),
            line_angle=float(p.line_angle or 0.0),
            line_curvature=float(p.line_curvature or 0.0),


            sensor_health={
                SensorType.ULTRASONIC: p.ultrasonic_cm is not None,
                SensorType.INFRARED: False,
                SensorType.CAMERA: p.camera_ok,
            },
            stale=stale,
        )